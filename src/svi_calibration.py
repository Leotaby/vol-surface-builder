"""
Stochastic Volatility Inspired (SVI) parameterization.

The SVI model (Gatheral, 2004) parameterizes total implied variance
w(k) as a function of log-moneyness k = ln(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where:
    a     = overall variance level
    b     = slope of the wings
    rho   = rotation / skew (-1 < rho < 1)
    m     = translation (shifts the minimum)
    sigma = curvature / ATM smile

This module provides:
    1. A simplified SVI generator tuned to produce realistic SPY-like surfaces
    2. A general SVI evaluation function for arbitrary parameters
    3. Parameter fitting via least-squares (for calibrating to market data)

The simplified generator is the default data source for the pipeline.
It produces surfaces that match the qualitative features of real equity
index vol surfaces: negative skew, steeper at short maturities, smile
curvature at the wings, and term structure flattening.

References:
    Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility derivatives.
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize

from . import config


# ════════════════════════════════════════════════════════════════════════
#  SVI EVALUATION
# ════════════════════════════════════════════════════════════════════════

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                       m: float, sigma: float) -> np.ndarray:
    """
    Compute SVI total implied variance w(k).

    Parameters
    ----------
    k : log-moneyness array, k = ln(K/F)
    a, b, rho, m, sigma : SVI parameters

    Returns
    -------
    np.ndarray : total implied variance w(k) = sigma_BS^2 * T
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_implied_vol(k: np.ndarray, T: float, a: float, b: float,
                    rho: float, m: float, sigma: float) -> np.ndarray:
    """
    Convert SVI total variance to implied volatility.

    sigma_BS = sqrt(w(k) / T)

    Returns NaN for any negative total variance (arbitrage violation).
    """
    w = svi_total_variance(k, a, b, rho, m, sigma)
    # total variance must be positive — if not, parameters are invalid
    w = np.where(w > 0, w, np.nan)
    return np.sqrt(w / T)


# ════════════════════════════════════════════════════════════════════════
#  SIMPLIFIED GENERATOR (tuned for SPY-like surfaces)
# ════════════════════════════════════════════════════════════════════════

def generate_svi_surface(
    S: float = 602.0,
    maturities: Optional[np.ndarray] = None,
    strikes: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Generate a realistic implied volatility surface using a simplified
    SVI-inspired parameterization.

    This isn't a full SVI fit — it's a reduced-form model that maps
    (log-moneyness, maturity) -> implied vol using three components:

        1. ATM level:    decays with maturity (term structure)
        2. Skew:         steeper at short maturities (gamma concentration)
        3. Curvature:    wings lift at all maturities (fat tails / smile)

    Each component has a "base" (long-run) value and a "short-maturity boost"
    that decays exponentially. Parameters are in config.py.

    Parameters
    ----------
    S : spot price (default: 602, approx SPY level)
    maturities : array of T values (years). Default: 10 points from 1wk to 2yr
    strikes : array of absolute strikes. Default: 2.5 spacing from 0.75S to 1.25S
    seed : random seed for micro-noise (default: config.SEED)

    Returns
    -------
    df : DataFrame with columns [strike, T, iv, moneyness, log_moneyness]
    S  : spot price used
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(config.SEED)

    if maturities is None:
        maturities = np.array([0.02, 0.04, 0.08, 0.17, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0])

    if strikes is None:
        strikes = np.arange(
            S * (1 - config.MONEYNESS_BOUND) - 10,
            S * (1 + config.MONEYNESS_BOUND) + 10,
            2.5
        )

    rows = []
    for T in maturities:
        # compute maturity-dependent SVI-like parameters
        atm_vol = config.SVI_ATM_BASE + config.SVI_ATM_DECAY * np.exp(-config.SVI_ATM_LAMBDA * T)
        skew_coeff = config.SVI_SKEW_SHORT * np.exp(-config.SVI_SKEW_LAMBDA * T) + config.SVI_SKEW_BASE
        smile_coeff = config.SVI_SMILE_SHORT * np.exp(-config.SVI_SMILE_LAMBDA * T) + config.SVI_SMILE_BASE

        for K in strikes:
            m = np.log(K / S)  # log-moneyness

            # simplified SVI: linear skew + quadratic curvature around ATM
            # this is equivalent to a second-order Taylor expansion of the
            # full SVI formula around the ATM point
            iv = atm_vol + skew_coeff * m + smile_coeff * m**2

            # add micro-noise for realism (real IV grids are never perfectly smooth)
            noise = np.random.normal(0, config.SVI_NOISE_STD)
            iv = np.clip(iv + noise, config.MIN_IV, config.MAX_IV)

            # moneyness filter — stay within configured bounds
            if abs(m) <= config.MONEYNESS_BOUND:
                rows.append({
                    "strike": K,
                    "T": T,
                    "iv": iv,
                    "moneyness": K / S,
                    "log_moneyness": m,
                })

    df = pd.DataFrame(rows)
    return df, S


# ════════════════════════════════════════════════════════════════════════
#  SVI CALIBRATION (fit to market data)
# ════════════════════════════════════════════════════════════════════════

def calibrate_svi_slice(
    log_moneyness: np.ndarray,
    market_iv: np.ndarray,
    T: float,
    initial_params: Optional[Dict] = None,
) -> Dict:
    """
    Fit SVI parameters to a single expiry slice of market IV data.

    Uses L-BFGS-B with parameter bounds to enforce basic constraints:
      - b > 0 (positive wing slope)
      - |rho| < 1 (correlation bound)
      - sigma > 0 (positive curvature)

    This is a basic fit — for production use, you'd want to add
    butterfly arbitrage constraints (Gatheral & Jacquier, 2014).

    Parameters
    ----------
    log_moneyness : array of k = ln(K/F) values
    market_iv : corresponding implied volatilities
    T : time to expiry (years)
    initial_params : optional starting point dict with keys a, b, rho, m, sigma

    Returns
    -------
    dict : fitted parameters {a, b, rho, m, sigma, rmse}
    """
    market_total_var = market_iv**2 * T  # convert IV to total variance

    if initial_params is None:
        # heuristic initial guess based on the data
        atm_var = np.interp(0.0, log_moneyness, market_total_var)
        initial_params = {
            "a": atm_var,
            "b": 0.1,
            "rho": -0.3,       # equity skew is typically negative
            "m": 0.0,
            "sigma": 0.1,
        }

    x0 = [initial_params["a"], initial_params["b"], initial_params["rho"],
          initial_params["m"], initial_params["sigma"]]

    # bounds: a is free, b > 0, -1 < rho < 1, m is free, sigma > 0
    bounds = [
        (None, None),     # a
        (1e-6, None),     # b
        (-0.999, 0.999),  # rho
        (None, None),     # m
        (1e-6, None),     # sigma
    ]

    def objective(params):
        a, b, rho, m, sigma = params
        model_var = svi_total_variance(log_moneyness, a, b, rho, m, sigma)
        residuals = model_var - market_total_var
        return np.sum(residuals**2)

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    a_fit, b_fit, rho_fit, m_fit, sigma_fit = result.x

    # compute RMSE in IV space for interpretability
    fitted_var = svi_total_variance(log_moneyness, a_fit, b_fit, rho_fit, m_fit, sigma_fit)
    fitted_iv = np.sqrt(np.maximum(fitted_var, 0) / T)
    rmse = np.sqrt(np.mean((fitted_iv - market_iv)**2))

    return {
        "a": a_fit,
        "b": b_fit,
        "rho": rho_fit,
        "m": m_fit,
        "sigma": sigma_fit,
        "rmse": rmse,
        "converged": result.success,
    }


def calibrate_svi_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit SVI parameters to each expiry slice in the dataframe.

    Parameters
    ----------
    df : DataFrame with columns [log_moneyness, iv, T]

    Returns
    -------
    DataFrame : one row per expiry with SVI parameters + RMSE
    """
    results = []
    for T_val in sorted(df["T"].unique()):
        slice_df = df[df["T"] == T_val].sort_values("log_moneyness")
        k = slice_df["log_moneyness"].values
        iv = slice_df["iv"].values

        if len(k) < 5:
            # not enough data points for a meaningful fit
            continue

        params = calibrate_svi_slice(k, iv, T_val)
        params["T"] = T_val
        results.append(params)

    return pd.DataFrame(results)
