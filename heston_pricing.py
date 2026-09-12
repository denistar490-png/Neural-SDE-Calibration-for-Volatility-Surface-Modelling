"""
Corrected semi-analytical Heston pricing using the P1/P2 formula.
The characteristic function below is for the full log-price log(S_T). It uses
the Little Heston Trap branch, with exp(-d*T) rather than exp(d*T), so the
formula remains numerically stable for the short maturities used in this
project.
"""

import numpy as np
from scipy.integrate import quad


INTEGRATION_LOWER_LIMIT = 1e-8
DEFAULT_EPSABS = 1e-8
DEFAULT_EPSREL = 1e-8
DEFAULT_LIMIT = 500


def heston_characteristic_function_corrected(
    u, S0, v0, kappa, theta, sigma, rho, T, r, q=0.0
):
    """Characteristic function of the full log-price log(S_T).
    For a complex argument u, let i denote sqrt(-1) and define
      beta = kappa - rho*sigma*i*u
      d = sqrt(beta**2 + sigma**2*(u**2 + i*u)).
    The square-root branch is selected so Re(d) >= 0. With
        g = (beta-d)/(beta+d),
    the Little Heston Trap uses exp(-d*T):
        C = i*u*(log(S0)+(r-q)*T)
            + kappa*theta/sigma**2 * ((beta-d)*T
            - 2*log((1-g*exp(-d*T))/(1-g)))
        D = (beta-d)/sigma**2
            * (1-exp(-d*T))/(1-g*exp(-d*T)).
    The logarithm is evaluated as two log1p terms instead of by forming the
    ratio first. This reduces cancellation and avoids an unnecessary complex
    logarithm branch jump.
    """
    if S0 <= 0.0:
        raise ValueError("S0 must be strictly positive")
    if T < 0.0:
        raise ValueError("T must be non-negative")
    if v0 < 0.0 or kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
        raise ValueError("Heston variance parameters are not admissible")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and 1")

    u_array = np.asarray(u, dtype=complex)
    i = 1j
    beta = kappa - rho * sigma * i * u_array
    d = np.sqrt(beta**2 + sigma**2 * (u_array**2 + i * u_array))
    d = np.where(np.real(d) < 0.0, -d, d)

    if T == 0.0:
        result = np.exp(i * u_array * np.log(S0))
    else:
        g = (beta - d) / (beta + d)
        exp_minus_dT = np.exp(-d * T)
        log_ratio = np.log1p(-g * exp_minus_dT) - np.log1p(-g)
        C = (
            i * u_array * (np.log(S0) + (r - q) * T)
            + (kappa * theta / sigma**2)
            * ((beta - d) * T - 2.0 * log_ratio)
        )
        D = (
            (beta - d) / sigma**2
            * (1.0 - exp_minus_dT)
            / (1.0 - g * exp_minus_dT)
        )
        result = np.exp(C + D * v0)

    if np.ndim(u) == 0:
        return complex(result)
    return result


def _heston_call_price_scalar(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    sigma,
    rho,
    q,
    epsabs,
    epsrel,
    limit,
):
    """Calculate one call price using the standard P1/P2 representation."""
    if K <= 0.0:
        raise ValueError("K must be strictly positive")
    if T <= 0.0:
        return max(S0 - K, 0.0)

    log_K = np.log(K)
    phi_minus_i = heston_characteristic_function_corrected(
        -1j, S0, v0, kappa, theta, sigma, rho, T, r, q
    )

    def p2_integrand(u):
        phi_u = heston_characteristic_function_corrected(
            u, S0, v0, kappa, theta, sigma, rho, T, r, q
        )
        return float(np.real(np.exp(-1j * u * log_K) * phi_u / (1j * u)))

    def p1_integrand(u):
        phi_u_minus_i = heston_characteristic_function_corrected(
            u - 1j, S0, v0, kappa, theta, sigma, rho, T, r, q
        )
        return float(
            np.real(
                np.exp(-1j * u * log_K)
                * phi_u_minus_i
                / (1j * u * phi_minus_i)
            )
        )

    p2_integral, _ = quad(
        p2_integrand,
        INTEGRATION_LOWER_LIMIT,
        np.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    p1_integral, _ = quad(
        p1_integrand,
        INTEGRATION_LOWER_LIMIT,
        np.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    P2 = 0.5 + p2_integral / np.pi
    P1 = 0.5 + p1_integral / np.pi
    return float(S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)


def heston_call_price_corrected(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    sigma,
    rho,
    q=0.0,
    epsabs=DEFAULT_EPSABS,
    epsrel=DEFAULT_EPSREL,
    limit=DEFAULT_LIMIT,
):
    """Return corrected Heston call prices for scalar or array strikes."""
    strikes = np.asarray(K, dtype=float)
    if strikes.ndim == 0:
        return _heston_call_price_scalar(
            S0, float(strikes), T, r, v0, kappa, theta, sigma, rho, q,
            epsabs, epsrel, limit
        )

    result = np.empty_like(strikes, dtype=float)
    for index in np.ndindex(strikes.shape):
        result[index] = _heston_call_price_scalar(
            S0, float(strikes[index]), T, r, v0, kappa, theta, sigma, rho, q,
            epsabs, epsrel, limit
        )
    return result


def heston_option_price_corrected(
    S0,
    K,
    T,
    r,
    params,
    option_type="call",
    q=0.0,
    epsabs=DEFAULT_EPSABS,
    epsrel=DEFAULT_EPSREL,
    limit=DEFAULT_LIMIT,
):
    """Return corrected Heston call or put prices using put-call parity."""
    v0, kappa, theta, sigma, rho = params
    call = heston_call_price_corrected(
        S0, K, T, r, v0, kappa, theta, sigma, rho, q=q,
        epsabs=epsabs, epsrel=epsrel, limit=limit
    )
    if option_type.lower() == "call":
        return call
    if option_type.lower() != "put":
        raise ValueError("option_type must be 'call' or 'put'")
    return call - S0 * np.exp(-q * np.asarray(T)) + np.asarray(K) * np.exp(-r * np.asarray(T))




heston_characteristic_function = heston_characteristic_function_corrected
heston_call_price = heston_call_price_corrected
heston_option_price = heston_option_price_corrected
