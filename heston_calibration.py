

import json

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.optimize import brentq, least_squares

from scipy.stats import norm

from heston_pricing import heston_option_price_corrected

S0 = 7165.08

RISK_FREE_RATE = 0.0359

DIVIDEND_YIELD = 0.0

MIN_MATURITY = 0.02

OPTION_TYPE = "put"

PRICER_EPSABS = 1e-7

PRICER_EPSREL = 1e-7

PRICER_LIMIT = 500

INVALID_IV_RESIDUAL = 1000.0

LOWER_BOUNDS = np.array([0.0001, 0.05, 0.0001, 0.001, -0.95], dtype=float)

UPPER_BOUNDS = np.array([0.50, 10.0, 0.50, 3.0, 0.0], dtype=float)

PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")

def load_fixed_parameters(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    return np.array([float(saved[name]) for name in PARAMETER_NAMES], dtype=float)

def load_calibration_data(path):
    """Reproduce the original put calibration filters."""
    data = pd.read_csv(path)
    data = data.dropna(subset=["type", "strike", "T", "mid", "IV"])
    data = data[
        (data["T"] >= MIN_MATURITY)
        & (data["mid"] > 0.0)
        & (data["IV"] > 0.0)
    ]
    data = data[data["type"].str.lower() == OPTION_TYPE].copy()
    data["moneyness"] = data["strike"] / S0
    data = data[(data["moneyness"] >= 0.80) & (data["moneyness"] <= 1.20)]
    data = data.sort_values(["T", "strike"]).reset_index(drop=True)
    data["representative_subset"] = False
    return data

def select_representative_subset(data, points_per_maturity=10):
    
    selected_indices = []
    for _, group in data.groupby("T", sort=True):
        ordered = group.sort_values("moneyness")
        count = min(points_per_maturity, len(ordered))
        positions = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
        selected_indices.extend(ordered.iloc[np.unique(positions)].index.tolist())

    selected_indices = sorted(set(selected_indices))
    data.loc[selected_indices, "representative_subset"] = True
    subset = data.loc[selected_indices].sort_values(["T", "moneyness"]).copy()
    return data, subset, selected_indices

def black_scholes_price(spot, strike, maturity, rate, volatility, option_type, q=0.0):
    if maturity <= 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    if volatility <= 0.0:
        discounted_spot = spot * np.exp(-q * maturity)
        discounted_strike = strike * np.exp(-rate * maturity)
        if option_type == "call":
            return max(discounted_spot - discounted_strike, 0.0)
        return max(discounted_strike - discounted_spot, 0.0)
    d1 = (
        np.log(spot / strike)
        + (rate - q + 0.5 * volatility ** 2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    if option_type == "call":
        return spot * np.exp(-q * maturity) * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    return strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * np.exp(-q * maturity) * norm.cdf(-d1)

def put_implied_volatility(price, spot, strike, maturity, rate, q=0.0):
    """Invert Black-Scholes for puts using discounted no-arbitrage bounds."""
    if not np.isfinite(price) or price <= 0.0 or maturity <= 0.0:
        return np.nan
    lower = max(strike * np.exp(-rate * maturity) - spot * np.exp(-q * maturity), 0.0)
    upper = strike * np.exp(-rate * maturity)
    if price <= lower or price >= upper:
        return np.nan

    def objective(volatility):
        return black_scholes_price(
            spot, strike, maturity, rate, volatility, "put", q
        ) - price

    try:
        return float(brentq(objective, 1e-6, 5.0, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan

def corrected_model_prices_and_ivs(params, data):
    strikes = data["strike"].to_numpy(float)
    maturities = data["T"].to_numpy(float)
    prices = np.array([
        heston_option_price_corrected(
            S0,
            strike,
            maturity,
            RISK_FREE_RATE,
            params,
            option_type=OPTION_TYPE,
            q=DIVIDEND_YIELD,
            epsabs=PRICER_EPSABS,
            epsrel=PRICER_EPSREL,
            limit=PRICER_LIMIT,
        )
        for strike, maturity in zip(strikes, maturities)
    ])
    model_iv = np.array([
        put_implied_volatility(
            price, S0, strike, maturity, RISK_FREE_RATE, DIVIDEND_YIELD
        )
        for price, strike, maturity in zip(prices, strikes, maturities)
    ])
    return prices, model_iv

def weighted_residuals(params, data, return_details=False):
    """Return weighted residuals, penalizing invalid IV."""
    prices, model_iv = corrected_model_prices_and_ivs(params, data)
    market_iv = data["IV"].to_numpy(float)
    moneyness = data["moneyness"].to_numpy(float)
    weights = 1.0 / (np.abs(moneyness - 1.0) + 0.05)
    valid = np.isfinite(model_iv) & np.isfinite(market_iv)
    errors = model_iv - market_iv
    residuals = np.empty(len(data), dtype=float)
    residuals[valid] = np.sqrt(weights[valid]) * errors[valid]
    residuals[~valid] = np.sqrt(weights[~valid]) * INVALID_IV_RESIDUAL

    if return_details:
        return residuals, {
            "prices": prices,
            "model_iv": model_iv,
            "market_iv": market_iv,
            "weights": weights,
            "errors": errors,
            "valid": valid,
            "invalid_count": int((~valid).sum()),
        }
    return residuals

def weighted_objective(residuals):
    return float(np.mean(np.asarray(residuals, dtype=float) ** 2))

def run_least_squares(data, initial_params, settings):
    def residual_function(params):
        return weighted_residuals(params, data)

    result = least_squares(
        residual_function,
        x0=initial_params,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        method=settings["method"],
        loss=settings["loss"],
        x_scale=settings["x_scale"],
        max_nfev=settings["max_nfev"],
        ftol=settings["ftol"],
        xtol=settings["xtol"],
        gtol=settings["gtol"],
        verbose=0,
    )
    return result, None

def fit_metrics(data, params):
    residuals, details = weighted_residuals(params, data, return_details=True)
    valid = details["valid"]
    errors = details["errors"]
    return {
        "residuals": residuals,
        "details": details,
        "valid_count": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(errors[valid] ** 2))) if valid.any() else np.nan,
        "mae": float(np.mean(np.abs(errors[valid]))) if valid.any() else np.nan,
        "max_abs_error": float(np.max(np.abs(errors[valid]))) if valid.any() else np.nan,
        "objective": weighted_objective(residuals),
    }

