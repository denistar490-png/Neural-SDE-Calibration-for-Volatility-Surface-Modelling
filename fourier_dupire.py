"""
Direct Fourier calculation of the Heston-associated local variance.
The implementation follows the one dimensional inversion identities in
Friz and Gerhold, equations (2.1),...,(2.4).  It works with the forward centred
log-price X_T = log(S_T / F_T), so the deterministic rate and dividend terms
are handled outside the Heston affine transform.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


PARAMETER_COUNT = 5
PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")


def _validate_inputs(spot: float, state: float, maturity: float, params) -> np.ndarray:
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and strictly positive")
    if not np.isfinite(state) or state <= 0.0:
        raise ValueError("state must be finite and strictly positive")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and strictly positive")

    values = np.asarray(params, dtype=float)
    if values.shape != (PARAMETER_COUNT,) or not np.isfinite(values).all():
        raise ValueError("params must contain five finite Heston parameters")
    v0, kappa, theta, sigma, rho = values
    if v0 < 0.0 or kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
        raise ValueError("Heston parameters are not admissible")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and 1")
    return values


def _affine_log_mgf_components(s: complex, maturity: float, params):
    """Return m(s,T), psi(s,T), psi_T(s,T), and the Trap denominator.

    For X_T = log(S_T/F_T), the affine transform is

        M(s,T) = E[exp(s X_T)] = exp(phi(s,T) + psi(s,T) v0).

    With a = kappa*theta, b = -kappa and c = sigma, the Riccati equations are

        phi_T = kappa*theta*psi,
        psi_T = 0.5*(s^2-s) + (rho*sigma*s-kappa)*psi
                 + 0.5*sigma^2*psi^2.

    The closed form below is the Little Heston Trap written for a Laplace
    argument s.  The square root is selected with Re(d) >= 0 and exp(-d*T)
    is used.  The logarithmic ratio is kept as two log1p terms; this avoids
    forming a singular complex quotient and its associated branch
    discontinuity.
    """
    v0, kappa, theta, sigma, rho = np.asarray(params, dtype=float)
    s = complex(s)
    beta = kappa - rho * sigma * s
    d = np.sqrt(beta * beta + sigma * sigma * s * (1.0 - s))
    if d.real < 0.0:
        d = -d

    g = (beta - d) / (beta + d)
    exp_minus_dT = np.exp(-d * maturity)
    trap_denominator = 1.0 - g * exp_minus_dT

    with np.errstate(all="ignore"):
        log_ratio = np.log1p(-g * exp_minus_dT) - np.log1p(-g)
        psi = (beta - d) / (sigma * sigma) * (-np.expm1(-d * maturity)) / trap_denominator
        phi = (kappa * theta / (sigma * sigma)) * (
            (beta - d) * maturity - 2.0 * log_ratio
        )
        log_mgf = phi + psi * v0
        psi_time_derivative = (
            0.5 * (s * s - s)
            + (rho * sigma * s - kappa) * psi
            + 0.5 * sigma * sigma * psi * psi
        )
        log_mgf_time_derivative = kappa * theta * psi + v0 * psi_time_derivative

    return log_mgf, psi, log_mgf_time_derivative, trap_denominator


def heston_forward_log_mgf(s: complex, maturity: float, params) -> complex:
    """Return log E[exp(s log(S_T/F_T))] for the fixed Heston parameters."""
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive")
    log_mgf, _, _, _ = _affine_log_mgf_components(s, maturity, params)
    return complex(log_mgf)


def heston_forward_log_mgf_time_derivative(
    s: complex, maturity: float, params
) -> complex:
    """Return the analytic maturity derivative of the affine log transform."""
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive")
    _, _, derivative, _ = _affine_log_mgf_components(s, maturity, params)
    return complex(derivative)


def _is_admissible_real_moment(s: float, maturity: float, params) -> bool:
    """Numerically test whether a real moment lies in the current strip."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            log_mgf, _, _, trap_denominator = _affine_log_mgf_components(
                complex(float(s), 0.0), maturity, params
            )
        if not np.isfinite(log_mgf.real) or not np.isfinite(log_mgf.imag):
            return False
        if not np.isfinite(trap_denominator.real) or not np.isfinite(trap_denominator.imag):
            return False
        if abs(trap_denominator) < 1e-10:
            return False
        # A real moment has a real positive transform and a tiny imaginary part
        
        if abs(log_mgf.imag) > 1e-7:
            return False
        return True
    except (RuntimeWarning, FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return False


def _find_moment_boundary(start: float, direction: float, maturity: float, params) -> float:
    """Find a finite moment boundary """
    admissible = float(start)
    step = 0.25
    for _ in range(100):
        candidate = admissible + direction * step
        if _is_admissible_real_moment(candidate, maturity, params):
            admissible = candidate
            step *= 1.5
            continue

        inadmissible = candidate
        for _ in range(80):
            midpoint = 0.5 * (admissible + inadmissible)
            if _is_admissible_real_moment(midpoint, maturity, params):
                admissible = midpoint
            else:
                inadmissible = midpoint
        return admissible

  
    
    return admissible


def admissible_moment_interval(maturity: float, params) -> tuple[float, float]:
    """Return the moment interval containing 0 and 1 at the given maturity."""
    values = np.asarray(params, dtype=float)
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive")
    if not _is_admissible_real_moment(0.0, maturity, values):
        raise ValueError("the zero moment was not numerically admissible")
    if not _is_admissible_real_moment(1.0, maturity, values):
        raise ValueError("the martingale moment was not numerically admissible")

    lower = _find_moment_boundary(0.0, -1.0, maturity, values)
    upper = _find_moment_boundary(1.0, 1.0, maturity, values)
    if not lower < 0.0 < 1.0 < upper:
        raise ValueError(
            f"invalid moment interval ({lower}, {upper}) at T={maturity}"
        )
    return float(lower), float(upper)


def _real_log_mgf_derivative(s: float, maturity: float, params) -> float:
    """Differentiate the transform in the real moment direction."""
    step = 1e-5 * max(1.0, abs(float(s)))
    right = heston_forward_log_mgf(float(s) + step, maturity, params)
    left = heston_forward_log_mgf(float(s) - step, maturity, params)
    derivative = (right - left) / (2.0 * step)
    if abs(derivative.imag) > 1e-6:
        raise ValueError("real moment derivative acquired a non-negligible imaginary part")
    return float(derivative.real)


def _saddle_shift(target: float, maturity: float, params, lower: float, upper: float) -> float:
    """Solve m_s(gamma,T)=target inside the moment interval."""
    
    
    left = lower + 0.01 * max(1.0, abs(lower))
    right = upper - 0.01 * max(1.0, abs(upper))
    left = min(left, -1e-5)
    right = max(right, 1.0 + 1e-5)
    grid = np.linspace(left, right, 401)
    values = []
    for point in grid:
        try:
            values.append(_real_log_mgf_derivative(float(point), maturity, params) - target)
        except (RuntimeWarning, FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            values.append(np.nan)
    for index in range(len(grid) - 1):
        left_value = values[index]
        right_value = values[index + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            return float(grid[index])
        if left_value * right_value < 0.0:
            return float(
                brentq(
                    lambda point: _real_log_mgf_derivative(point, maturity, params) - target,
                    float(grid[index]),
                    float(grid[index + 1]),
                    xtol=1e-10,
                    rtol=1e-12,
                    maxiter=100,
                )
            )
    raise ValueError(f"could not locate a saddle contour for target log-moneyness {target}")


def admissible_contour_shifts(
    maturity: float, params, target_log_moneyness: float | None = None
) -> tuple[float, float, float]:
    """Choose three shifts strictly inside the checked moment strip.
    When a target log-moneyness is supplied, the middle shift solves the
    Friz--Gerhold saddle equation m_s(gamma,T)=k.  The two neighboring shifts
    stay in the same pole-free segment of the moment strip.
    """
    lower, upper = admissible_moment_interval(maturity, params)
    if target_log_moneyness is None:
        available = upper - 1.0
        if not np.isfinite(available) or available <= 1e-8:
            raise ValueError("no usable contour interval to the right of the call poles")
        usable_width = min(available, 1.0)
        return tuple(1.0 + fraction * usable_width for fraction in (0.25, 0.50, 0.75))

    saddle = _saddle_shift(float(target_log_moneyness), maturity, params, lower, upper)
    if saddle < 0.0:
        segment_lower, segment_upper = lower, -1e-8
    elif saddle < 1.0:
        segment_lower, segment_upper = 1e-8, 1.0 - 1e-8
    else:
        segment_lower, segment_upper = 1.0 + 1e-8, upper
    distance = min(saddle - segment_lower, segment_upper - saddle)
    if not np.isfinite(distance) or distance <= 1e-10:
        raise ValueError("saddle contour is too close to a pole or moment boundary")
    step = min(0.5, 0.25 * distance)
    return float(saddle - step), float(saddle), float(saddle + step)


def _validate_contour_shift(contour_shift: float, maturity: float, params) -> tuple[float, float]:
    lower, upper = admissible_moment_interval(maturity, params)
    if (
        not np.isfinite(contour_shift)
        or not (lower < contour_shift < upper)
        or abs(contour_shift) <= 1e-8
        or abs(contour_shift - 1.0) <= 1e-8
    ):
        raise ValueError(
            "contour_shift must be strictly inside the moment interval and "
            f"upper moment boundary ({lower}, {upper})"
        )
    return lower, upper


def _integrate_real(integrand, tolerance, limit):
    # A quadrature failure invalidates Dupire, do not turn it into a surface value.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = quad(integrand, 0., np.inf, epsabs=tolerance, epsrel=tolerance,
                      limit=limit, full_output=1)
    if len(result) > 3:
        raise FloatingPointError('Fourier integral did not converge: '+str(result[3]))
    return float(result[0])


def heston_local_variance_fourier(
    spot,
    state,
    maturity,
    rate,
    params,
    q=0.0,
    integration_tolerance=1e-9,
    integration_limit=500,
    contour_shift=None,
):
    """Return the Heston-associated Dupire local variance at (T, state).
    The forward log-moneyness is k = log(state / F_T), with
    F_T = spot * exp((rate-q)*T).  With M(s,T) the transform of
    X_T = log(S_T/F_T), the two real half-line integrals are
        D = int Re[exp(-k s) M(s,T)] du,
        N = int Re[exp(-k s) M(s,T) m_T(s,T)/(s(s-1))] du,
    on s = gamma + i u.  The Dupire ratio is local_variance = 2*N/D.
    """
    values = _validate_inputs(float(spot), float(state), float(maturity), params)
    if not np.isfinite(rate) or not np.isfinite(q):
        raise ValueError("rate and q must be finite")
    if not np.isfinite(integration_tolerance) or integration_tolerance <= 0.0:
        raise ValueError("integration_tolerance must be strictly positive")
    if int(integration_limit) != integration_limit or integration_limit <= 0:
        raise ValueError("integration_limit must be a positive integer")

    maturity = float(maturity)
    integration_limit = int(integration_limit)
    forward = float(spot) * np.exp((float(rate) - float(q)) * maturity)
    log_moneyness = float(np.log(float(state) / forward))
    selected_shifts = admissible_contour_shifts(maturity, values, log_moneyness)
    if contour_shift is None:
        contour_shift = selected_shifts[1]
    contour_shift = float(contour_shift)
    lower_moment, upper_moment = _validate_contour_shift(contour_shift, maturity, values)

    reference_log_scale, _, _, _ = _affine_log_mgf_components(
        complex(contour_shift, 0.0), maturity, values
    )
    reference_log_scale = float(np.real(reference_log_scale - log_moneyness * contour_shift))

    def denominator_integrand(u: float) -> float:
        s = complex(contour_shift, u)
        log_mgf = heston_forward_log_mgf(s, maturity, values)
        return float(np.real(np.exp(-log_moneyness * s + log_mgf - reference_log_scale)))

    def numerator_integrand(u: float) -> float:
        s = complex(contour_shift, u)
        log_mgf, _, log_mgf_time_derivative, _ = _affine_log_mgf_components(
            s, maturity, values
        )
        return float(
            np.real(
                np.exp(-log_moneyness * s + log_mgf)
                * np.exp(-reference_log_scale)
                * log_mgf_time_derivative
                / (s * (s - 1.0))
            )
        )

    denominator = _integrate_real(denominator_integrand, float(integration_tolerance), integration_limit)
    numerator = _integrate_real(numerator_integrand, float(integration_tolerance), integration_limit)
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(numerator) or numerator <= 0:
        raise FloatingPointError('Invalid Fourier-Dupire numerator or denominator')
    return 2.0*numerator/denominator
