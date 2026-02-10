"""
Surface construction: from scattered (strike, maturity, IV) triples
to a smooth, regular grid suitable for 3D plotting and analysis.

The challenge: real option chains don't have the same strikes across
expiries. Short-dated options might have strikes every $1 while LEAPS
have $5 or $10 spacing. Some strikes are liquid (tight bid-ask) and
others are basically indicative quotes with no real volume.

The pipeline:
    1. Take the cleaned scattered data from data_feed
    2. Define a regular (K, T) grid
    3. Interpolate IV onto the grid using scipy.griddata
    4. Handle boundary NaNs with nearest-neighbor fallback
    5. Optionally smooth the result with a Gaussian filter

The output is a set of numpy arrays (K_grid, T_grid, K_mesh, T_mesh, IV_mesh)
that can be passed directly to the visualization module.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional

from . import config


def build_surface(
    df: pd.DataFrame,
    n_k: int = None,
    n_t: int = None,
    method: str = None,
    smooth_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate scattered IV data onto a regular 2D grid.

    Parameters
    ----------
    df : DataFrame with at least columns [strike, T, iv]
    n_k : number of grid points along strike axis (default: config.GRID_K_POINTS)
    n_t : number of grid points along maturity axis (default: config.GRID_T_POINTS)
    method : interpolation method — "cubic", "linear", or "nearest"
             (default: config.INTERPOLATION_METHOD)
    smooth_sigma : if provided, apply Gaussian smoothing with this sigma.
                   useful for noisy live data. None = no smoothing.

    Returns
    -------
    K_grid : 1D array of strike values (length n_k)
    T_grid : 1D array of maturity values (length n_t)
    K_mesh : 2D meshgrid of strikes (n_t x n_k)
    T_mesh : 2D meshgrid of maturities (n_t x n_k)
    IV_mesh : 2D array of interpolated implied vols (n_t x n_k)
    """
    if n_k is None:
        n_k = config.GRID_K_POINTS
    if n_t is None:
        n_t = config.GRID_T_POINTS
    if method is None:
        method = config.INTERPOLATION_METHOD

    strikes = df["strike"].values
    maturities = df["T"].values
    ivs = df["iv"].values

    # grid bounds: use percentiles to avoid extreme outliers pulling the range
    K_min, K_max = np.percentile(strikes, [2, 98])
    T_min = maturities.min()
    T_max = maturities.max()

    # avoid degenerate grids
    if K_max - K_min < 1.0:
        K_min -= 5.0
        K_max += 5.0
    if T_max - T_min < 0.001:
        T_max = T_min + 0.01

    K_grid = np.linspace(K_min, K_max, n_k)
    T_grid = np.linspace(T_min, T_max, n_t)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    # primary interpolation
    IV_mesh = griddata(
        points=(strikes, maturities),
        values=ivs,
        xi=(K_mesh, T_mesh),
        method=method,
    )

    # cubic interpolation can produce NaN at grid edges where there's
    # no data to form a full cubic patch. fill these with nearest-neighbor
    # which always produces a value (even if it's less smooth).
    nan_mask = np.isnan(IV_mesh)
    if nan_mask.any():
        IV_nearest = griddata(
            points=(strikes, maturities),
            values=ivs,
            xi=(K_mesh, T_mesh),
            method="nearest",
        )
        IV_mesh[nan_mask] = IV_nearest[nan_mask]

    # optional smoothing — helps with noisy live data
    if smooth_sigma is not None and smooth_sigma > 0:
        IV_mesh = gaussian_filter(IV_mesh, sigma=smooth_sigma)

    return K_grid, T_grid, K_mesh, T_mesh, IV_mesh


def extract_skew_slices(
    df: pd.DataFrame,
    target_maturities: list = None,
) -> dict:
    """
    Extract IV vs strike data for specific maturities.

    Used for the 2D skew chart. Finds the closest available maturity
    to each target and returns the corresponding data.

    Parameters
    ----------
    df : DataFrame with columns [strike, T, iv]
    target_maturities : list of T values to extract. Default: config.SKEW_TARGET_MATURITIES

    Returns
    -------
    dict : {T_value: DataFrame subset} for each matched maturity
    """
    if target_maturities is None:
        target_maturities = config.SKEW_TARGET_MATURITIES

    available_T = sorted(df["T"].unique())
    slices = {}

    for target_T in target_maturities:
        closest_T = min(available_T, key=lambda x: abs(x - target_T))
        if closest_T not in slices:
            subset = df[df["T"] == closest_T].sort_values("strike").copy()
            slices[closest_T] = subset

    return slices


def compute_surface_statistics(
    df: pd.DataFrame,
    S: float,
) -> dict:
    """
    Compute summary statistics for the vol surface.

    Useful for quick diagnostics and for the methodology doc.

    Parameters
    ----------
    df : DataFrame with columns [strike, T, iv, log_moneyness]
    S : spot price

    Returns
    -------
    dict with keys:
        n_points      : total data points
        n_expiries    : number of distinct maturities
        strike_range  : (min, max)
        T_range       : (min, max)
        iv_range      : (min, max)
        atm_iv_mean   : average IV for near-ATM strikes
        skew_25d      : approximate 25-delta risk reversal (put IV - call IV)
    """
    stats = {
        "n_points": len(df),
        "n_expiries": df["T"].nunique(),
        "strike_range": (df["strike"].min(), df["strike"].max()),
        "T_range": (df["T"].min(), df["T"].max()),
        "iv_range": (df["iv"].min(), df["iv"].max()),
    }

    # ATM IV: average IV for strikes within 1% of spot
    atm_mask = df["moneyness"].between(0.99, 1.01) if "moneyness" in df.columns else pd.Series([False] * len(df))
    if atm_mask.any():
        stats["atm_iv_mean"] = df.loc[atm_mask, "iv"].mean()
    else:
        stats["atm_iv_mean"] = np.nan

    # rough 25-delta skew proxy: compare IV at 90% moneyness vs 110%
    if "moneyness" in df.columns:
        put_side = df[df["moneyness"].between(0.89, 0.91)]
        call_side = df[df["moneyness"].between(1.09, 1.11)]
        if len(put_side) > 0 and len(call_side) > 0:
            stats["skew_25d_proxy"] = put_side["iv"].mean() - call_side["iv"].mean()
        else:
            stats["skew_25d_proxy"] = np.nan
    else:
        stats["skew_25d_proxy"] = np.nan

    return stats
