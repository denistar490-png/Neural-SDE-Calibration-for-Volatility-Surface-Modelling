"""
Piecewise neural leverage model for the Neural LSV thesis stage.
The stochastic volatility parameters are immutable buffers.  Seven separate
networks represent the seven left closed, right open maturity intervals.  The
explicit unit leverage mode bypasses the networks exactly and is the pure SABR
baseline used by the local validation task.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable

import torch
from torch import nn


MATURITY_DAYS = (8, 15, 22, 30, 60, 90, 180)
MATURITY_BOUNDARIES_YEARS = (0.0,) + tuple(day / 365.0 for day in MATURITY_DAYS)
NETWORK_WIDTH = 64
LEAKY_RELU_SLOPE = 0.2
INITIAL_WEIGHT_STD = 0.05


class LeverageSubnetwork(nn.Module):
    """The 1-64-64-64-64-1 architecture used for one maturity interval."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, NETWORK_WIDTH),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Linear(NETWORK_WIDTH, NETWORK_WIDTH),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Linear(NETWORK_WIDTH, NETWORK_WIDTH),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Linear(NETWORK_WIDTH, NETWORK_WIDTH),
            nn.Tanh(),
            nn.Linear(NETWORK_WIDTH, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, mean=0.0, std=INITIAL_WEIGHT_STD)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point tensor")
        return self.network(x.unsqueeze(-1)).squeeze(-1)


class NeuralLSVModel(nn.Module):
    """Seven-slice leverage model with frozen beta=1 SABR parameters."""

    def __init__(
        self,
        alpha0: float,
        nu: float,
        rho: float,
        *,
        beta: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not alpha0 > 0.0:
            raise ValueError("alpha0 must be strictly positive")
        if not nu >= 0.0:
            raise ValueError("nu must be non-negative")
        if not abs(rho) < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")
        if beta != 1.0:
            raise ValueError("This implementation requires beta exactly equal to one")

        factory = {"dtype": dtype, "device": device}
        # Keep the calibrated reference values outside autograd. This permits a
        # new SABR calibration while still detecting accidental buffer updates.
        self.frozen_sabr_parameters = dict(alpha0=float(alpha0), nu=float(nu),
                                           rho=float(rho), beta=float(beta))
        self.register_buffer("alpha0", torch.tensor(float(alpha0), **factory))
        self.register_buffer("nu", torch.tensor(float(nu), **factory))
        self.register_buffer("rho", torch.tensor(float(rho), **factory))
        self.register_buffer("beta", torch.tensor(float(beta), **factory))
        self.register_buffer(
            "maturity_boundaries",
            torch.tensor(MATURITY_BOUNDARIES_YEARS, **factory),
        )
        self.leverage_networks = nn.ModuleList(LeverageSubnetwork() for _ in range(7))
        self.to(device=device, dtype=dtype)
        self._trainable_slice: int | None = None
        self.freeze_all()

    @staticmethod
    def _validate_slice_index(slice_index: int) -> int:
        if isinstance(slice_index, bool) or not isinstance(slice_index, int):
            raise TypeError("slice_index must be an integer from 1 through 7")
        if slice_index < 1 or slice_index > 7:
            raise ValueError("slice_index must be in 1,...,7")
        return slice_index

    def slice_index(self, time_years: float | torch.Tensor) -> int:
        """Return a one-based network index for a scalar time before 180 days."""
        if isinstance(time_years, torch.Tensor):
            if time_years.numel() != 1:
                raise ValueError("time_years must be scalar")
            value = float(time_years.detach().cpu())
        else:
            value = float(time_years)
        if not 0.0 <= value < MATURITY_BOUNDARIES_YEARS[-1]:
            raise ValueError("time_years must lie in [0, 180/365)")
        # bisect_right sends an exact internal boundary to the next interval.
        return 1 + bisect_right(MATURITY_BOUNDARIES_YEARS[1:-1], value)

    def leverage(
        self,
        time_years: float | torch.Tensor,
        x: torch.Tensor,
        *,
        force_unit_leverage: bool = False,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if force_unit_leverage:
            return torch.ones_like(x)
        index = self.slice_index(time_years) - 1
        # The paper-style parameterisation is used literally: no positivity
        # transform, absolute value, floor, cap, or clipping is applied.
        return 1.0 + self.leverage_networks[index](x)

    def set_trainable_slice(self, slice_index: int) -> None:
        selected = self._validate_slice_index(slice_index)
        for index, network in enumerate(self.leverage_networks, start=1):
            requires_gradient = index == selected
            for parameter in network.parameters():
                parameter.requires_grad_(requires_gradient)
                parameter.grad = None
        self._trainable_slice = selected

    def freeze_slice(self, slice_index: int) -> None:
        selected = self._validate_slice_index(slice_index)
        for parameter in self.leverage_networks[selected - 1].parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if self._trainable_slice == selected:
            self._trainable_slice = None

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._trainable_slice = None

    def get_trainable_slice(self) -> int | None:
        active = []
        for index, network in enumerate(self.leverage_networks, start=1):
            flags = {parameter.requires_grad for parameter in network.parameters()}
            if len(flags) != 1:
                raise RuntimeError(f"Slice {index} has inconsistent gradient flags")
            if flags == {True}:
                active.append(index)
        if len(active) > 1:
            raise RuntimeError("More than one leverage slice is trainable")
        observed = active[0] if active else None
        if observed != self._trainable_slice:
            raise RuntimeError("Stored and observed trainable slices disagree")
        return observed

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


__all__ = [
    "INITIAL_WEIGHT_STD",
    "LEAKY_RELU_SLOPE",
    "MATURITY_BOUNDARIES_YEARS",
    "MATURITY_DAYS",
    "NETWORK_WIDTH",
    "LeverageSubnetwork",
    "NeuralLSVModel",
]
