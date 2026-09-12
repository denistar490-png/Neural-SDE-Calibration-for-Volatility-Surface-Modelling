"""The 48 start global beta=1 Hagan calibration used in the thesis."""
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sabr_formula import BETA,sabr_beta1_implied_volatility,sabr_beta1_option_price
S0=np.float64(7165.08)
RISK_FREE_RATE=np.float64(.0359)
DIVIDEND_YIELD=np.float64(0)
EXPECTED_OPTIONS=203
LOWER_BOUNDS=np.array([1e-4,0.,-.999])
UPPER_BOUNDS=np.array([2.,10.,.999])
INVALID_RESIDUAL=np.float64(1000.)

class CalibrationFailure(RuntimeError):
    pass

def calibrate(target):
    """Select the lowest unweighted IV SSE among successful starts."""
    forward = target.forward.to_numpy(np.float64)
    strike = target.strike.to_numpy(np.float64)
    maturity = target.maturity_years.to_numpy(np.float64)
    target_iv = target.heston_implied_volatility.to_numpy(np.float64)
    def residual(p):
        try:
            iv = np.asarray(sabr_beta1_implied_volatility(forward, strike, maturity,
                                                        float(p[0]), float(p[1]), float(p[2])), dtype=np.float64)
            if np.any(~np.isfinite(iv)) or np.any(iv <= 0):
                raise ValueError('Invalid SABR volatility')
            return iv-target_iv
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(target_iv.shape, INVALID_RESIDUAL, dtype=np.float64)
    best, best_error = None, np.inf
    starts, _ = make_starting_points(target)
    for start in starts:
        result = least_squares(residual, start, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                               method='trf', loss='linear', x_scale='jac',
                               ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
        error = float(np.dot(result.fun, result.fun))
        if result.success and np.isfinite(error) and error < min(best_error, INVALID_RESIDUAL**2*EXPECTED_OPTIONS):
            best, best_error = result.x.copy(), error
    if best is None:
        raise CalibrationFailure('No SABR calibration start converged')
    return dict(alpha0=float(best[0]), nu=float(best[1]), rho=float(best[2]), beta=1.0)

def heston_otm_prices(frame: pd.DataFrame) -> np.ndarray:
    maturity = frame["maturity_years"].to_numpy(dtype=np.float64)
    strike = frame["strike"].to_numpy(dtype=np.float64)
    call = frame["heston_call_price"].to_numpy(dtype=np.float64)
    put = call - S0 * np.exp(-DIVIDEND_YIELD * maturity) + strike * np.exp(
        -RISK_FREE_RATE * maturity
    )
    result = np.where(frame["option_type"].eq("call"), call, put).astype(np.float64)
    if np.any((~np.isfinite(result)) | (result <= 0.0)):
        raise CalibrationFailure("Parity-consistent Heston OTM price is invalid")
    return result

def make_starting_points(target: pd.DataFrame) -> tuple[list[np.ndarray], dict[str, Any]]:
    near_atm = (
        target.assign(abs_k=np.abs(target["forward_log_moneyness"]))
        .sort_values(["maturity_days", "abs_k"], kind="mergesort")
        .groupby("maturity_days", sort=True)
        .head(1)
    )
    atm_levels = near_atm["heston_implied_volatility"].to_numpy(dtype=np.float64)
    median_atm = float(np.median(atm_levels))
    alpha_candidates = median_atm * np.array([0.75, 1.0, 1.25, 1.50], dtype=np.float64)
    nu_candidates = np.array([0.25, 1.50, 4.00], dtype=np.float64)
    rho_candidates = np.array([-0.80, -0.35, 0.00, 0.35], dtype=np.float64)
    starts = [
        np.array([alpha, nu, rho], dtype=np.float64)
        for alpha in alpha_candidates
        for nu in nu_candidates
        for rho in rho_candidates
    ]
    if len(starts) < 32:
        raise CalibrationFailure("Fewer than 32 deterministic starts were constructed")
    for start in starts:
        if np.any(start < LOWER_BOUNDS) or np.any(start > UPPER_BOUNDS):
            raise CalibrationFailure(f"Deterministic start is outside bounds: {start}")
    design = {
        "observed_near_atm_iv_by_maturity": {
            str(int(row.maturity_days)): float(row.heston_implied_volatility)
            for row in near_atm.itertuples(index=False)
        },
        "median_near_atm_iv": median_atm,
        "alpha0_candidates": alpha_candidates,
        "nu_candidates": nu_candidates,
        "rho_candidates": rho_candidates,
        "start_count": len(starts),
        "construction": "Cartesian product of four ATM-based alpha0 levels, three nu levels, and four rho levels; no random starts.",
    }
    return starts, design

def build_fit_results(target: pd.DataFrame, parameters: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "maturity_days": target["maturity_days"].astype(np.int64),
            "maturity_years": target["maturity_years"].astype(np.float64),
            "maturity_source": target["maturity_source"].astype(str),
            "option_type": target["option_type"].astype(str),
            "strike": target["strike"].astype(np.float64),
            "forward": target["forward"].astype(np.float64),
            "forward_log_moneyness": target["forward_log_moneyness"].astype(np.float64),
            "heston_call_price": target["heston_call_price"].astype(np.float64),
            "heston_price": heston_otm_prices(target),
            "heston_implied_volatility": target["heston_implied_volatility"].astype(
                np.float64
            ),
        }
    )
    for column in target.columns:
        if (
            "moneyness" in column.lower()
            and column != "forward_log_moneyness"
            and column not in result.columns
        ):
            result[f"original_{column}"] = target[column].to_numpy()
    result["sabr_alpha0"] = float(parameters[0])
    result["sabr_nu"] = float(parameters[1])
    result["sabr_rho"] = float(parameters[2])
    result["sabr_implied_volatility"] = sabr_beta1_implied_volatility(
        result["forward"].to_numpy(dtype=np.float64),
        result["strike"].to_numpy(dtype=np.float64),
        result["maturity_years"].to_numpy(dtype=np.float64),
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
    )
    result["sabr_black_scholes_price"] = sabr_beta1_option_price(
        float(S0),
        result["strike"].to_numpy(dtype=np.float64),
        result["maturity_years"].to_numpy(dtype=np.float64),
        float(RISK_FREE_RATE),
        float(DIVIDEND_YIELD),
        result["option_type"].to_numpy(),
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
    )
    return result
