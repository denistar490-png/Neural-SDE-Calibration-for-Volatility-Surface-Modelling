"""
Hagan lognormal SABR approximation for beta equal to one.
The following code implements one globally parameterised forward model
    dF_t = alpha_t F_t dW_t^(1)
    d alpha_t = nu alpha_t dW_t^(2)
    d<W^(1),W^(2)>_t = rho dt,
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.special import ndtr


BETA = np.float64(1.0)
SMALL_Z_THRESHOLD = np.float64(1e-4)


def _broadcast_float64(
    names: Iterable[str], values: Iterable[object]
) -> tuple[list[np.ndarray], bool]:
    names = list(names)
    raw = [np.asarray(value, dtype=np.float64) for value in values]
    scalar_output = all(array.ndim == 0 for array in raw)
    try:
        broadcast = list(np.broadcast_arrays(*raw))
    except ValueError as error:
        raise ValueError(
            f"Inputs are not broadcast-compatible: {', '.join(names)}"
        ) from error
    return [np.asarray(array, dtype=np.float64) for array in broadcast], scalar_output


def _first_invalid_message(
    mask: np.ndarray,
    message: str,
    names: list[str],
    arrays: list[np.ndarray],
) -> str:
    flat_index = int(np.flatnonzero(mask.reshape(-1))[0])
    coordinates = {
        name: float(array.reshape(-1)[flat_index]) for name, array in zip(names, arrays)
    }
    return (
        f"{message}; {int(np.count_nonzero(mask))} invalid broadcast element(s). "
        f"First invalid values: {coordinates}"
    )


def _reject(
    mask: np.ndarray,
    message: str,
    names: list[str],
    arrays: list[np.ndarray],
) -> None:
    if np.any(mask):
        raise ValueError(_first_invalid_message(mask, message, names, arrays))


def _finish(result: np.ndarray, scalar_output: bool):
    result = np.asarray(result, dtype=np.float64)
    if scalar_output:
        return float(result.reshape(-1)[0])
    return result


def sabr_beta1_implied_volatility(
    forward,
    strike,
    maturity_years,
    alpha0,
    nu,
    rho,
):
    """Return Hagan beta=1 lognormal implied volatility.
    For z, the formula uses
      z = (nu/alpha0) log(F/K),
        x(z) = log((sqrt(1-2 rho z+z^2)+z-rho)/(1-rho)).
    Direct evaluation of z/x(z) is avoided near zero.  For |z| <= 1e-4,
    the stable expansion
        z/x(z) = 1 - rho*z/2 + (2-3*rho^2)*z^2/12 + O(z^3)
    is used.  Exact ATM and exact nu=0 limits are handled explicitly.
    """

    names = ["forward", "strike", "maturity_years", "alpha0", "nu", "rho"]
    arrays, scalar_output = _broadcast_float64(
        names, [forward, strike, maturity_years, alpha0, nu, rho]
    )
    forward_array, strike_array, maturity, alpha, vol_of_vol, correlation = arrays

    _reject(
        ~np.isfinite(np.stack([array.reshape(-1) for array in arrays], axis=0)).all(axis=0).reshape(forward_array.shape),
        "Every SABR input must be finite",
        names,
        arrays,
    )
    _reject(forward_array <= 0.0, "forward must be strictly positive", names, arrays)
    _reject(strike_array <= 0.0, "strike must be strictly positive", names, arrays)
    _reject(maturity < 0.0, "maturity_years must be non-negative", names, arrays)
    _reject(alpha <= 0.0, "alpha0 must be strictly positive", names, arrays)
    _reject(vol_of_vol < 0.0, "nu must be non-negative", names, arrays)
    _reject(np.abs(correlation) >= 1.0, "abs(rho) must be strictly below one", names, arrays)

    maturity_coefficient = (
        correlation * vol_of_vol * alpha / np.float64(4.0)
        + (np.float64(2.0) - np.float64(3.0) * correlation**2)
        * vol_of_vol**2
        / np.float64(24.0)
    )
    maturity_correction = np.float64(1.0) + maturity_coefficient * maturity
    _reject(
        (~np.isfinite(maturity_correction)) | (maturity_correction <= 0.0),
        "SABR maturity correction must be finite and strictly positive",
        names,
        arrays,
    )

    result = np.empty(forward_array.shape, dtype=np.float64)
    zero_nu = vol_of_vol == 0.0
    result[zero_nu] = alpha[zero_nu]

    active = ~zero_nu
    if np.any(active):
        log_forward_over_strike = np.log(
            forward_array[active] / strike_array[active]
        ).astype(np.float64, copy=False)
        z = (
            vol_of_vol[active] / alpha[active] * log_forward_over_strike
        ).astype(np.float64, copy=False)
        if not np.isfinite(z).all():
            active_indices = np.flatnonzero(active.reshape(-1))
            local_bad = int(np.flatnonzero(~np.isfinite(z))[0])
            global_bad = int(active_indices[local_bad])
            raise ValueError(
                "SABR z must be finite; first invalid flattened broadcast index "
                f"is {global_bad}"
            )

        ratio = np.empty(z.shape, dtype=np.float64)
        exact_atm = log_forward_over_strike == 0.0
        ratio[exact_atm] = 1.0
        small = (~exact_atm) & (np.abs(z) <= SMALL_Z_THRESHOLD)
        if np.any(small):
            z_small = z[small]
            rho_small = correlation[active][small]
            ratio[small] = (
                np.float64(1.0)
                - np.float64(0.5) * rho_small * z_small
                + (np.float64(2.0) - np.float64(3.0) * rho_small**2)
                * z_small**2
                / np.float64(12.0)
            )

        regular = (~exact_atm) & (~small)
        if np.any(regular):
            z_regular = z[regular]
            rho_regular = correlation[active][regular]
            x_z = np.empty(z_regular.shape, dtype=np.float64)
            zero_correlation = rho_regular == 0.0
            # At rho=0, x(z)=asinh(z) exactly.  This identity avoids the
            # cancellation in log(sqrt(1+z^2)+z) and preserves the required
            # odd symmetry of x(z) to float64 precision.
            x_z[zero_correlation] = np.arcsinh(z_regular[zero_correlation])
            generic = ~zero_correlation
            if np.any(generic):
                z_generic = z_regular[generic]
                rho_generic = rho_regular[generic]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    square_root_argument = (
                        np.float64(1.0)
                        - np.float64(2.0) * rho_generic * z_generic
                        + z_generic**2
                    )
                invalid_square_root = (
                    (~np.isfinite(square_root_argument))
                    | (square_root_argument <= 0.0)
                )
                if np.any(invalid_square_root):
                    raise ValueError(
                        "SABR square-root argument must be finite and strictly positive"
                    )
                square_root = np.sqrt(square_root_argument, dtype=np.float64)
                logarithm_argument = (
                    square_root + z_generic - rho_generic
                ) / (np.float64(1.0) - rho_generic)
                if np.any(
                    (~np.isfinite(logarithm_argument))
                    | (logarithm_argument <= 0.0)
                ):
                    raise ValueError(
                        "SABR logarithm argument must be finite and strictly positive"
                    )
                x_z[generic] = np.log(logarithm_argument)
            if np.any((~np.isfinite(x_z)) | (x_z == 0.0)):
                raise ValueError("SABR x(z) must be finite and nonzero away from ATM")
            ratio[regular] = z_regular / x_z

        active_result = (
            alpha[active] * ratio * maturity_correction[active]
        ).astype(np.float64, copy=False)
        result[active] = active_result

    if np.any((~np.isfinite(result)) | (result <= 0.0)):
        raise ValueError("SABR implied volatility must be finite and strictly positive")
    return _finish(result, scalar_output)


def _black_scholes_option_price_from_volatility(
    spot,
    strike,
    maturity_years,
    r,
    q,
    volatility,
    option_type,
):
    names = ["spot", "strike", "maturity_years", "r", "q", "volatility"]
    arrays, scalar_numeric = _broadcast_float64(
        names, [spot, strike, maturity_years, r, q, volatility]
    )
    spot_array, strike_array, maturity, rate, dividend, sigma = arrays
    option_raw = np.asarray(option_type)
    scalar_output = scalar_numeric and option_raw.ndim == 0
    try:
        option_array = np.broadcast_to(option_raw, spot_array.shape)
    except ValueError as error:
        raise ValueError("option_type is not broadcast-compatible") from error
    option_array = np.char.lower(option_array.astype(str))

    stacked = np.stack([array.reshape(-1) for array in arrays], axis=0)
    _reject(
        (~np.isfinite(stacked).all(axis=0)).reshape(spot_array.shape),
        "Every Black-Scholes input must be finite",
        names,
        arrays,
    )
    _reject(spot_array <= 0.0, "spot must be strictly positive", names, arrays)
    _reject(strike_array <= 0.0, "strike must be strictly positive", names, arrays)
    _reject(maturity < 0.0, "maturity_years must be non-negative", names, arrays)
    _reject(sigma <= 0.0, "volatility must be strictly positive", names, arrays)
    invalid_type = (option_array != "call") & (option_array != "put")
    if np.any(invalid_type):
        first = str(option_array.reshape(-1)[int(np.flatnonzero(invalid_type.reshape(-1))[0])])
        raise ValueError(f"option_type must be 'call' or 'put'; first invalid value is {first!r}")

    discounted_spot = spot_array * np.exp(-dividend * maturity)
    discounted_strike = strike_array * np.exp(-rate * maturity)
    call_intrinsic = np.maximum(discounted_spot - discounted_strike, 0.0)
    put_intrinsic = np.maximum(discounted_strike - discounted_spot, 0.0)
    result = np.empty(spot_array.shape, dtype=np.float64)
    expired = maturity == 0.0
    result[expired] = np.where(
        option_array[expired] == "call",
        call_intrinsic[expired],
        put_intrinsic[expired],
    )

    live = ~expired
    if np.any(live):
        root_time = np.sqrt(maturity[live], dtype=np.float64)
        d1 = (
            np.log(spot_array[live] / strike_array[live])
            + (rate[live] - dividend[live] + 0.5 * sigma[live] ** 2)
            * maturity[live]
        ) / (sigma[live] * root_time)
        d2 = d1 - sigma[live] * root_time
        call = (
            discounted_spot[live] * ndtr(d1)
            - discounted_strike[live] * ndtr(d2)
        )
        put = (
            discounted_strike[live] * ndtr(-d2)
            - discounted_spot[live] * ndtr(-d1)
        )
        result[live] = np.where(option_array[live] == "call", call, put)

    if not np.isfinite(result).all():
        raise ValueError("Black-Scholes option price must be finite")
    return _finish(result, scalar_output)


def sabr_beta1_option_price(
    spot,
    strike,
    maturity_years,
    r,
    q,
    option_type,
    alpha0,
    nu,
    rho,
):
    """Return a call or put price using SABR IV and Black-Scholes."""
    numeric_names = ["spot", "strike", "maturity_years", "r", "q", "alpha0", "nu", "rho"]
    numeric, scalar_numeric = _broadcast_float64(
        numeric_names, [spot, strike, maturity_years, r, q, alpha0, nu, rho]
    )
    spot_array, strike_array, maturity, rate, dividend, alpha, vol_of_vol, correlation = numeric
    forward = spot_array * np.exp((rate - dividend) * maturity)
    if np.any((~np.isfinite(forward)) | (forward <= 0.0)):
        raise ValueError("Constructed forward must be finite and strictly positive")
    volatility = sabr_beta1_implied_volatility(
        forward, strike_array, maturity, alpha, vol_of_vol, correlation
    )
    price = _black_scholes_option_price_from_volatility(
        spot_array,
        strike_array,
        maturity,
        rate,
        dividend,
        volatility,
        option_type,
    )
    option_scalar = np.asarray(option_type).ndim == 0
    if scalar_numeric and option_scalar:
        return float(np.asarray(price, dtype=np.float64).reshape(-1)[0])
    return np.asarray(price, dtype=np.float64)


def sabr_beta1_call_price(
    spot,
    strike,
    maturity_years,
    r,
    q,
    alpha0,
    nu,
    rho,
):
    """Return a SABR European call price."""
    return sabr_beta1_option_price(
        spot,
        strike,
        maturity_years,
        r,
        q,
        "call",
        alpha0,
        nu,
        rho,
    )


__all__ = [
    "BETA",
    "SMALL_Z_THRESHOLD",
    "sabr_beta1_implied_volatility",
    "sabr_beta1_call_price",
    "sabr_beta1_option_price",
]
