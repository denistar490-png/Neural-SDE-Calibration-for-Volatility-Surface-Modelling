"""Neural/SABR log-Euler simulation and Black-Scholes hedge."""
import math
from typing import Any
import numpy as np
import pandas as pd
import torch
from neural_model import MATURITY_DAYS, NeuralLSVModel
from control_variate import (discounted_option_payoffs, discounted_option_payoffs_from_log,
    discounted_state_increment_from_log, discounted_state_bs_delta, discounted_state_bs_delta_from_log)
S0=7165.08
RISK_FREE_RATE=.0359
DIVIDEND_YIELD=0.0

def _device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA torch devices are supported")
    return resolved

def generate_standard_normals(
    n_steps: int,
    n_paths: int,
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if n_steps <= 0 or n_paths <= 1:
        raise ValueError("n_steps must be positive and n_paths must exceed one")
    target_device = _device(device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (n_steps, n_paths, 2),
        generator=generator,
        dtype=dtype,
        device=target_device,
    )

def correlated_brownian_drivers(
    independent_normals: torch.Tensor,
    rho: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if independent_normals.ndim < 1 or independent_normals.shape[-1] != 2:
        raise ValueError("independent_normals must have final dimension two")
    if torch.any(~torch.isfinite(independent_normals)):
        raise ValueError("independent_normals must be finite")
    correlation = torch.as_tensor(
        rho, dtype=independent_normals.dtype, device=independent_normals.device
    )
    if correlation.numel() != 1 or not bool(torch.abs(correlation) < 1.0):
        raise ValueError("rho must be scalar and lie strictly between -1 and 1")
    z1 = independent_normals[..., 0]
    z2 = independent_normals[..., 1]
    correlated = correlation * z1 + torch.sqrt(1.0 - correlation.square()) * z2
    return z1, correlated

def exact_alpha_update(
    alpha: torch.Tensor,
    nu: torch.Tensor | float,
    dt: float,
    z_alpha: torch.Tensor,
) -> torch.Tensor:
    """Exact one-step update of d alpha_t = nu alpha_t dB_t."""
    volatility_of_volatility = torch.as_tensor(
        nu, dtype=alpha.dtype, device=alpha.device
    )
    return alpha * torch.exp(
        -0.5 * volatility_of_volatility.square() * dt
        + volatility_of_volatility * math.sqrt(dt) * z_alpha
    )

def _one_maturity_targets(
    targets: pd.DataFrame,
    maturity_index: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    rate: float,
) -> dict[str, Any]:
    group = targets.loc[targets["maturity_index"] == maturity_index].copy()
    group = group.sort_values("strike", kind="mergesort").reset_index(drop=True)
    if len(group) != 29:
        raise ValueError(f"Maturity index {maturity_index} must contain exactly 29 options")
    maturity_days = int(group["maturity_days"].iloc[0])
    expected_days = MATURITY_DAYS[maturity_index - 1]
    if maturity_days != expected_days or group["maturity_days"].nunique() != 1:
        raise ValueError("Maturity index and day count disagree")
    maturity_years = maturity_days / 365.0
    strike = torch.tensor(group["strike"].to_numpy(np.float64), dtype=dtype, device=device)
    discounted_strike = strike * math.exp(-rate * maturity_years)
    option_type = group["option_type"].astype(str).str.lower().tolist()
    if not set(option_type).issubset({"call", "put"}):
        raise ValueError("Option types must be calls or puts")
    return {
        "frame": group,
        "maturity_days": maturity_days,
        "maturity_years": maturity_years,
        "strike": strike,
        "discounted_strike": discounted_strike,
        "option_type": option_type,
    }

def _prepare_normals(
    *,
    n_steps: int,
    n_paths: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    external_normals: torch.Tensor | np.ndarray | None,
) -> torch.Tensor:
    if external_normals is None:
        return generate_standard_normals(
            n_steps, n_paths, seed=seed, dtype=dtype, device=device
        )
    supplied = torch.as_tensor(external_normals, dtype=dtype, device=device)
    expected = (n_steps, n_paths, 2)
    if tuple(supplied.shape) != expected:
        raise ValueError(f"external_normals shape must be {expected}, got {tuple(supplied.shape)}")
    if torch.any(~torch.isfinite(supplied)):
        raise ValueError("external_normals must be finite")
    return supplied

def _state_check(name: str, value: torch.Tensor, *, strictly_positive: bool = False) -> None:
    if torch.any(~torch.isfinite(value)):
        raise FloatingPointError(f"Non-finite {name} state detected")
    if strictly_positive and torch.any(value <= 0.0):
        raise FloatingPointError(f"Non-positive {name} state detected")

def _sample_statistics(samples: torch.Tensor) -> dict[str, torch.Tensor]:
    variance = torch.var(samples, dim=0, unbiased=True)
    mean = torch.mean(samples, dim=0)
    standard_error = torch.sqrt(variance / samples.shape[0])
    return {"mean": mean, "variance": variance, "standard_error": standard_error}


def simulate(model, targets, n_paths, steps_per_day, seed, *, maturity_index=None,
             unit_leverage=False, numerical_mode="log_state_robust",
             hedge_mode="alpha_only", external_normals=None, collect_states=False):            
    if numerical_mode not in ("legacy", "log_state_robust"):
        raise ValueError("Unknown numerical mode")
    if hedge_mode not in ("running_vol", "alpha_only"):
        raise ValueError("Unknown hedge mode")
    if n_paths < 2 or steps_per_day < 1:
        raise ValueError("Need at least two paths and one step per day")
    dtype, device = model.alpha0.dtype, model.alpha0.device
    indices = [maturity_index] if maturity_index is not None else list(range(1, 8))
    groups = {i: _one_maturity_targets(targets, i, dtype=dtype, device=device,
                                      rate=RISK_FREE_RATE) for i in indices}
    end_steps = {i: g['maturity_days']*steps_per_day for i, g in groups.items()}
    dt = 1.0/(365.0*steps_per_day)
    root_dt = math.sqrt(dt)
    normals = _prepare_normals(n_steps=max(end_steps.values()), n_paths=n_paths,
                              seed=seed, dtype=dtype, device=device, external_normals=external_normals)
    z1, zb = correlated_brownian_drivers(normals, model.rho)
    x = torch.zeros(n_paths, dtype=dtype, device=device)
    alpha = torch.full_like(x, float(model.alpha0.detach().cpu()))
    log_spot = x.new_tensor(math.log(S0))
    hedges = {i: torch.zeros((n_paths, 29), dtype=dtype, device=device) for i in indices}
    results = {}
    state_samples = {i: [] for i in range(1, 8)} if collect_states else None
    for step in range(max(end_steps.values())):
        t = step*dt
        if collect_states:
            state_samples[model.slice_index(t)].append(x.detach()[::32].cpu().numpy().copy())
        leverage = model.leverage(t, x, force_unit_leverage=unit_leverage)
        volatility = alpha*leverage
        _state_check('alpha', alpha, strictly_positive=True)
        _state_check('diffusion', volatility)
        x_next = x - 0.5*volatility.square()*dt + volatility*root_dt*z1[step]
        _state_check('updated log state', x_next)
        with torch.no_grad():
            hedge_vol = alpha.detach() if hedge_mode == 'alpha_only' else volatility.detach()
            if numerical_mode == 'legacy':
                y = S0*torch.exp(x)
                y_next = S0*torch.exp(x_next)
                _state_check('discounted spot', y_next, strictly_positive=True)
                dy = y_next-y
            else:
                log_y = log_spot+x.detach()
                log_y_next = log_spot+x_next.detach()
                dy, _ = discounted_state_increment_from_log(log_y, log_y_next)
            for i, group in groups.items():
                if step >= end_steps[i]:
                    continue
                tau = group['maturity_years']-t
                if numerical_mode == 'legacy':
                    delta = discounted_state_bs_delta(y, group['discounted_strike'], tau, hedge_vol, group['option_type'])
                else:
                    delta = discounted_state_bs_delta_from_log(log_y, group['discounted_strike'], tau, hedge_vol, group['option_type'])
                _state_check('hedge delta', delta)
                hedges[i].add_(delta*dy.reshape(-1, 1))
        alpha = exact_alpha_update(alpha, model.nu, dt, zb[step])
        _state_check('updated alpha', alpha, strictly_positive=True)
        x = x_next
        for i, group in groups.items():
            if step+1 != end_steps[i]:
                continue
            if numerical_mode == 'legacy':
                payoff = discounted_option_payoffs(S0*torch.exp(x), group['discounted_strike'], group['option_type'])
            else:
                payoff = discounted_option_payoffs_from_log(log_spot+x, group['discounted_strike'], group['option_type'])
            _state_check('payoff', payoff)
            adjusted = payoff-hedges[i].detach()
            results[i] = {'frame': group['frame'], 'raw': _sample_statistics(payoff),
                          'variance_reduced': _sample_statistics(adjusted)}
    return results, state_samples
