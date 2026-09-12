"""Black-Scholes coding part used by the other parts of the project"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_price(S0, K, T, r, vol, option_type="call", q=0.0):
    if T <= 0 or vol <= 0:
        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def implied_volatility(price, S0, K, T, r, option_type="call", q=0.0):
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    if option_type == "call":
        intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper = S0 * np.exp(-q * T)
    else:
        intrinsic = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
        upper = K * np.exp(-r * T)
    if price <= intrinsic + 1e-10 or price >= upper:
        return np.nan

    def objective(vol):
        return black_scholes_price(S0, K, T, r, vol, option_type, q) - price

    try:
        return brentq(objective, 1e-4, 5.0, maxiter=100)
    except ValueError:
        return np.nan
