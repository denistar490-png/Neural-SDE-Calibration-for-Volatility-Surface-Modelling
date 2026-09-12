"""Fixed Heston grid, OTM prices, and inverse-Vega neural training targets."""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from heston_pricing import heston_option_price_corrected
S0=7165.08
RISK_FREE_RATE=.0359
DIVIDEND_YIELD=0.0
DEFAULT_EPSABS=DEFAULT_EPSREL=1e-8
PRICER_LIMIT=750
EVALUATION_MATURITY_DAYS=np.array([8,15,22,30,60,90,180])
EVALUATION_FORWARD_LOG_MONEYNESS=np.arange(-.20,.0800001,.01)
MATURITY_DAYS=EVALUATION_MATURITY_DAYS.tolist()
EXPECTED_ROWS=203

def price_grid(parameters):
    """Price the fixed 7-by-29 grid, then invert OTM prices to implied volatility."""
    grid = build_evaluation_grid()
    calls, ivs, types = [], [], []
    for row in grid.itertuples(index=False):
        call = price_call(S0, row.strike, row.maturity_years, parameters, 1e-8, 1e-8, 750)
        kind = 'call' if row.strike >= row.forward else 'put'
        price = call if kind == 'call' else call-S0+row.strike*np.exp(-RISK_FREE_RATE*row.maturity_years)
        iv = implied_volatility_from_price(price, S0, row.strike, row.maturity_years, kind)
        if not np.isfinite(call) or not np.isfinite(iv) or iv <= 0:
            raise ValueError('Invalid Heston price or implied volatility')
        calls.append(call); ivs.append(iv); types.append(kind)
    grid['heston_call_price'] = calls
    grid['heston_implied_volatility'] = ivs
    grid['option_type'] = types
    return grid

def price_call(S0_value, strike, maturity, params, epsabs, epsrel, limit):
    return float(
        heston_option_price_corrected(
            S0_value,
            float(strike),
            float(maturity),
            RISK_FREE_RATE,
            params,
            option_type="call",
            q=DIVIDEND_YIELD,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
    )

def black_scholes_price(spot, strike, maturity, volatility, option_type):
    if maturity <= 0.0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if volatility <= 0.0:
        discounted_spot = spot * np.exp(-DIVIDEND_YIELD * maturity)
        discounted_strike = strike * np.exp(-RISK_FREE_RATE * maturity)
        if option_type == "call":
            return max(discounted_spot - discounted_strike, 0.0)
        return max(discounted_strike - discounted_spot, 0.0)

    d1 = (
        np.log(spot / strike)
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * volatility ** 2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    if option_type == "call":
        return (
            spot * np.exp(-DIVIDEND_YIELD * maturity) * norm.cdf(d1)
            - strike * np.exp(-RISK_FREE_RATE * maturity) * norm.cdf(d2)
        )
    return (
        strike * np.exp(-RISK_FREE_RATE * maturity) * norm.cdf(-d2)
        - spot * np.exp(-DIVIDEND_YIELD * maturity) * norm.cdf(-d1)
    )

def implied_volatility_from_price(price, spot, strike, maturity, option_type):
    if not np.isfinite(price) or price <= 0.0 or maturity <= 0.0:
        return np.nan
    lower, upper = option_bounds(spot, strike, maturity, option_type)
    if price <= lower or price >= upper:
        return np.nan

    def objective(volatility):
        return black_scholes_price(spot, strike, maturity, volatility, option_type) - price

    try:
        return float(brentq(objective, 1e-8, 5.0, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan

def black_scholes_vega(spot, strike, maturity, volatility):
    if not np.isfinite(volatility) or volatility <= 0.0 or maturity <= 0.0:
        return np.nan
    d1 = (
        np.log(spot / strike)
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * volatility ** 2) * maturity
    ) / (volatility * np.sqrt(maturity))
    return float(spot * np.exp(-DIVIDEND_YIELD * maturity) * norm.pdf(d1) * np.sqrt(maturity))

def build_evaluation_grid():
    rows = []
    for maturity_days in EVALUATION_MATURITY_DAYS:
        maturity_years = maturity_days / 365.0
        forward = S0 * np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * maturity_years)
        source = (
            "market_supported_maturity"
            if maturity_days <= 30
            else "heston_extrapolation_maturity"
        )
        for log_moneyness in EVALUATION_FORWARD_LOG_MONEYNESS:
            strike = forward * np.exp(log_moneyness)
            rows.append(
                {
                    "maturity_days": int(maturity_days),
                    "maturity_years": maturity_years,
                    "maturity_source": source,
                    "inside_common_market_region": bool(-0.14 <= log_moneyness <= 0.04),
                    "spot": S0,
                    "risk_free_rate": RISK_FREE_RATE,
                    "dividend_yield": DIVIDEND_YIELD,
                    "forward": forward,
                    "forward_log_moneyness": float(log_moneyness),
                    "strike": float(strike),
                }
            )
    return pd.DataFrame(rows)
def prepare_targets(heston: pd.DataFrame, sabr_fit: pd.DataFrame) -> pd.DataFrame:
    if DIVIDEND_YIELD != 0.0:
        raise AssertionError("The discounted-state representation requires q == 0.0")
    keys = ["maturity_days", "strike"]
    if heston.duplicated(keys).any() or sabr_fit.duplicated(keys).any():
        raise ValueError("Target matching keys must be unique")
    heston_columns = heston[
        ["maturity_days", "strike", "heston_implied_volatility"]
    ].rename(columns={"heston_implied_volatility": "raw_heston_iv"})
    fit_columns = sabr_fit[
        [
            "maturity_days",
            "maturity_years",
            "strike",
            "option_type",
            "heston_price",
            "heston_implied_volatility",
        ]
    ]
    merged = fit_columns.merge(heston_columns, on=keys, how="inner", validate="one_to_one")
    if len(merged) != EXPECTED_ROWS:
        raise ValueError("Heston and SABR fit inputs did not match one-to-one")
    if not np.allclose(
        merged["heston_implied_volatility"], merged["raw_heston_iv"], rtol=0.0, atol=2e-15
    ):
        raise ValueError("Heston implied volatilities differ between input files")

    maturity_index = {day: index for index, day in enumerate(MATURITY_DAYS, start=1)}
    maturity_days = merged["maturity_days"].to_numpy(np.int64)
    maturity_years = maturity_days.astype(np.float64) / 365.0
    strike = merged["strike"].to_numpy(np.float64)
    sigma = merged["heston_implied_volatility"].to_numpy(np.float64)
    forward = S0 * np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * maturity_years)
    log_moneyness = np.log(strike / forward)
    root_time = np.sqrt(maturity_years)
    d1 = (
        np.log(S0 / strike)
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * sigma**2) * maturity_years
    ) / (sigma * root_time)
    normal_density = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    vega = S0 * np.exp(-DIVIDEND_YIELD * maturity_years) * normal_density * root_time
    if np.any((~np.isfinite(vega)) | (vega <= 0.0)):
        bad = np.flatnonzero((~np.isfinite(vega)) | (vega <= 0.0)).tolist()
        raise ValueError(f"Invalid unmodified Black-Scholes target Vega at rows {bad}")
    raw_weight = 1.0 / vega

    result = pd.DataFrame(
        {
            "maturity_days": maturity_days,
            "maturity_years": maturity_years,
            "maturity_index": [maturity_index[int(day)] for day in maturity_days],
            "strike": strike,
            "option_type": merged["option_type"].astype(str).str.lower(),
            "heston_target_price": merged["heston_price"].to_numpy(np.float64),
            "heston_target_implied_volatility": sigma,
            "forward": forward,
            "forward_log_moneyness": log_moneyness,
            "black_scholes_target_vega": vega,
            "unnormalized_vega_weight": raw_weight,
        }
    )
    result["normalized_vega_weight"] = result["unnormalized_vega_weight"] / result.groupby(
        "maturity_days"
    )["unnormalized_vega_weight"].transform("sum")
    result = result.sort_values(
        ["maturity_index", "strike"], kind="mergesort"
    ).reset_index(drop=True)
    weight_sums = result.groupby("maturity_days")["normalized_vega_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(), 1.0, rtol=0.0, atol=5e-16):
        raise ValueError(f"Normalized weights do not sum to one: {weight_sums.to_dict()}")
    numeric = result.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Prepared target table contains a non-finite numeric value")
    if (result["normalized_vega_weight"] <= 0.0).any():
        raise ValueError("Prepared normalized weights must be strictly positive")
    return result

def option_bounds(spot, strike, maturity, option_type):
    discounted_spot = spot * np.exp(-DIVIDEND_YIELD * maturity)
    discounted_strike = strike * np.exp(-RISK_FREE_RATE * maturity)
    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike

