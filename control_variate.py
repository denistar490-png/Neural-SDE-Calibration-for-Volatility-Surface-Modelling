"""
Analytic Black-Scholes delta control variate for discounted Neural LSV paths.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / torch.sqrt(value.new_tensor(2.0))))


def _call_mask(
    option_type: str | Sequence[str] | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(option_type, str):
        lowered = option_type.lower()
        if lowered not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return torch.tensor(lowered == "call", dtype=torch.bool, device=device)
    if isinstance(option_type, torch.Tensor):
        if option_type.dtype != torch.bool:
            raise TypeError("Tensor option_type must be a Boolean call mask")
        return option_type.to(device=device)
    lowered = [str(value).lower() for value in option_type]
    invalid = [value for value in lowered if value not in {"call", "put"}]
    if invalid:
        raise ValueError(f"Invalid option type: {invalid[0]!r}")
    return torch.tensor([value == "call" for value in lowered], dtype=torch.bool, device=device)


def discounted_state_bs_delta(
    discounted_spot: torch.Tensor,
    discounted_strike: torch.Tensor,
    tau: float | torch.Tensor,
    instantaneous_volatility: torch.Tensor,
    option_type: str | Sequence[str] | torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes delta in the zero-rate discounted-state representation.

    The absolute value is applied only to the volatility entering d1.  It does
    not alter the leverage value or the SDE diffusion coefficient.
    """
    y = discounted_spot
    k_bar = discounted_strike.to(dtype=y.dtype, device=y.device)
    remaining = torch.as_tensor(tau, dtype=y.dtype, device=y.device)
    sigma = torch.abs(instantaneous_volatility)
    if torch.any(~torch.isfinite(y)) or torch.any(y <= 0.0):
        raise ValueError("discounted_spot must be finite and strictly positive")
    if torch.any(~torch.isfinite(k_bar)) or torch.any(k_bar <= 0.0):
        raise ValueError("discounted_strike must be finite and strictly positive")
    if torch.any(~torch.isfinite(remaining)) or torch.any(remaining <= 0.0):
        raise ValueError("tau must be finite and strictly positive")
    if torch.any(~torch.isfinite(sigma)) or torch.any(sigma <= 0.0):
        raise ValueError("Black-Scholes hedge volatility must be finite and positive")

    y_column = y.reshape(-1, 1)
    sigma_column = sigma.reshape(-1, 1)
    strike_row = k_bar.reshape(1, -1)
    root_tau = torch.sqrt(remaining)
    d1 = (
        torch.log(y_column / strike_row)
        + 0.5 * sigma_column.square() * remaining
    ) / (sigma_column * root_tau)
    call_delta = standard_normal_cdf(d1)
    mask = _call_mask(option_type, device=y.device)
    if mask.ndim == 0:
        return call_delta if bool(mask) else call_delta - 1.0
    if mask.numel() != strike_row.numel():
        raise ValueError("option_type count must equal discounted-strike count")
    return torch.where(mask.reshape(1, -1), call_delta, call_delta - 1.0)


def discounted_state_bs_delta_from_log(
    log_discounted_spot: torch.Tensor,
    discounted_strike: torch.Tensor,
    tau: float | torch.Tensor,
    instantaneous_volatility: torch.Tensor,
    option_type: str | Sequence[str] | torch.Tensor,
) -> torch.Tensor:
    
    log_y = log_discounted_spot
    k_bar = discounted_strike.to(dtype=log_y.dtype, device=log_y.device)
    remaining = torch.as_tensor(tau, dtype=log_y.dtype, device=log_y.device)
    sigma = torch.abs(instantaneous_volatility)
    if torch.any(~torch.isfinite(log_y)):
        raise ValueError("log_discounted_spot must be finite")
    if torch.any(~torch.isfinite(k_bar)) or torch.any(k_bar <= 0.0):
        raise ValueError("discounted_strike must be finite and strictly positive")
    if torch.any(~torch.isfinite(remaining)) or torch.any(remaining <= 0.0):
        raise ValueError("tau must be finite and strictly positive")
    if torch.any(~torch.isfinite(sigma)) or torch.any(sigma <= 0.0):
        raise ValueError("Black-Scholes hedge volatility must be finite and positive")

    log_y_column = log_y.reshape(-1, 1)
    sigma_column = sigma.reshape(-1, 1)
    log_strike_row = torch.log(k_bar).reshape(1, -1)
    root_tau = torch.sqrt(remaining)
    d1 = (
        log_y_column
        - log_strike_row
        + 0.5 * sigma_column.square() * remaining
    ) / (sigma_column * root_tau)
    call_delta = standard_normal_cdf(d1)
    mask = _call_mask(option_type, device=log_y.device)
    if mask.ndim == 0:
        return call_delta if bool(mask) else call_delta - 1.0
    if mask.numel() != log_strike_row.numel():
        raise ValueError("option_type count must equal discounted-strike count")
    return torch.where(mask.reshape(1, -1), call_delta, call_delta - 1.0)


def ordinary_spot_bs_delta(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_years: float | torch.Tensor,
    maturity_years: float | torch.Tensor,
    rate: float,
    dividend_yield: float,
    volatility: torch.Tensor,
    option_type: str | Sequence[str] | torch.Tensor,
) -> torch.Tensor:
    """Ordinary Black-Scholes spot delta for validation."""
    current = torch.as_tensor(time_years, dtype=spot.dtype, device=spot.device)
    maturity = torch.as_tensor(maturity_years, dtype=spot.dtype, device=spot.device)
    tau = maturity - current
    sigma = torch.abs(volatility)
    if torch.any(tau <= 0.0):
        raise ValueError("maturity_years must exceed time_years")
    if torch.any(spot <= 0.0) or torch.any(strike <= 0.0) or torch.any(sigma <= 0.0):
        raise ValueError("spot, strike, and volatility must be positive")
    spot_column = spot.reshape(-1, 1)
    sigma_column = sigma.reshape(-1, 1)
    strike_row = strike.to(dtype=spot.dtype, device=spot.device).reshape(1, -1)
    d1 = (
        torch.log(spot_column / strike_row)
        + (rate - dividend_yield + 0.5 * sigma_column.square()) * tau
    ) / (sigma_column * torch.sqrt(tau))
    dividend_discount = torch.exp(
        -torch.as_tensor(dividend_yield, dtype=spot.dtype, device=spot.device) * tau
    )
    call_delta = dividend_discount * standard_normal_cdf(d1)
    mask = _call_mask(option_type, device=spot.device)
    if mask.ndim == 0:
        return call_delta if bool(mask) else call_delta - torch.exp(
            -torch.as_tensor(dividend_yield, dtype=spot.dtype, device=spot.device) * tau
        )
    return torch.where(mask.reshape(1, -1), call_delta, call_delta - dividend_discount)


def discounted_option_payoffs(
    terminal_discounted_spot: torch.Tensor,
    discounted_strike: torch.Tensor,
    option_type: Sequence[str] | torch.Tensor,
) -> torch.Tensor:
    y = terminal_discounted_spot.reshape(-1, 1)
    k_bar = discounted_strike.to(dtype=y.dtype, device=y.device).reshape(1, -1)
    call = torch.relu(y - k_bar)
    put = torch.relu(k_bar - y)
    mask = _call_mask(option_type, device=y.device)
    if mask.ndim == 0:
        return call if bool(mask) else put
    return torch.where(mask.reshape(1, -1), call, put)


def discounted_option_payoffs_from_log(
    log_terminal_discounted_spot: torch.Tensor,
    discounted_strike: torch.Tensor,
    option_type: Sequence[str] | torch.Tensor,
) -> torch.Tensor:
    """ European payoffs evaluated from ``log(Y_T)``.

    The relative log-moneyness is masked before ``expm1`` is evaluated.  This
    prevents overflow in an inactive payoff branch while retaining the payoff
    gradient wherever the option is in the money.
    """
    log_y = log_terminal_discounted_spot.reshape(-1, 1)
    k_bar = discounted_strike.to(dtype=log_y.dtype, device=log_y.device).reshape(1, -1)
    if torch.any(~torch.isfinite(log_y)):
        raise ValueError("log_terminal_discounted_spot must be finite")
    if torch.any(~torch.isfinite(k_bar)) or torch.any(k_bar <= 0.0):
        raise ValueError("discounted_strike must be finite and strictly positive")

    relative_log_spot = log_y - torch.log(k_bar)
    zeros = torch.zeros_like(relative_log_spot)
    call_itm = relative_log_spot > 0.0
    put_itm = relative_log_spot < 0.0
    call_relative = torch.where(call_itm, relative_log_spot, zeros)
    put_relative = torch.where(put_itm, relative_log_spot, zeros)
    call = torch.where(call_itm, k_bar * torch.expm1(call_relative), zeros)
    put = torch.where(put_itm, -k_bar * torch.expm1(put_relative), zeros)
    mask = _call_mask(option_type, device=log_y.device)
    if mask.ndim == 0:
        return call if bool(mask) else put
    return torch.where(mask.reshape(1, -1), call, put)


def discounted_state_increment_from_log(
    log_discounted_spot_current: torch.Tensor,
    log_discounted_spot_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Y_next - Y_current`` and a below-precision event mask.
    Scaling by the larger log spot keeps both relative exponentials in
    ``[0, 1]``.  If both true spots lie below the smallest positive value of
    the working dtype, their absolute difference is also below representable
    precision and zero is returned explicitly, no positive floor is used.
    """
    current, following = torch.broadcast_tensors(
        log_discounted_spot_current, log_discounted_spot_next
    )
    if torch.any(~torch.isfinite(current)) or torch.any(~torch.isfinite(following)):
        raise ValueError("Log discounted spots must be finite")
    maximum = torch.maximum(current, following)
    zero = maximum.new_tensor(0.0)
    smallest_positive = torch.nextafter(zero, maximum.new_tensor(1.0))
    log_smallest_positive = torch.log(smallest_positive)
    below_absolute_precision = maximum < log_smallest_positive

    scale = torch.exp(maximum)
    if torch.any(torch.isinf(scale)):
        raise FloatingPointError("Derived discounted-spot increment overflow")
    relative_difference = (
        torch.exp(following - maximum) - torch.exp(current - maximum)
    )
    increment = scale * relative_difference
    increment = torch.where(below_absolute_precision, torch.zeros_like(increment), increment)
    if torch.any(~torch.isfinite(increment)):
        raise FloatingPointError("Non-finite discounted-spot increment")
    return increment, below_absolute_precision


def fixed_vega_weighted_price_loss(
    model_prices: torch.Tensor,
    target_prices: torch.Tensor,
    normalized_inverse_vega_weights: torch.Tensor,
) -> torch.Tensor:
    
    target = target_prices.to(dtype=model_prices.dtype, device=model_prices.device)
    weights = normalized_inverse_vega_weights.to(
        dtype=model_prices.dtype, device=model_prices.device
    )
    if model_prices.ndim != 1 or target.shape != model_prices.shape or weights.shape != model_prices.shape:
        raise ValueError("prices, targets, and weights must be equal one-dimensional arrays")
    if torch.any(~torch.isfinite(weights)) or torch.any(weights <= 0.0):
        raise ValueError("Fixed inverse-Vega weights must be finite and positive")
    
    
    tolerance = max(1e-12, 8 * torch.finfo(weights.dtype).eps)
    if not torch.allclose(weights.sum(), weights.new_tensor(1.0), rtol=0, atol=tolerance):
        raise ValueError("Fixed inverse-Vega weights must sum to one")
    return torch.sum(weights * (model_prices - target).square())


__all__ = [
    "discounted_option_payoffs",
    "discounted_option_payoffs_from_log",
    "discounted_state_increment_from_log",
    "discounted_state_bs_delta",
    "discounted_state_bs_delta_from_log",
    "fixed_vega_weighted_price_loss",
    "ordinary_spot_bs_delta",
    "standard_normal_cdf",
]
