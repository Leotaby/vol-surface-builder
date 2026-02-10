"""
Option chain data retrieval and cleaning pipeline.

Supports two modes:
    1. Live: pull from yfinance (requires internet + market hours for fresh data)
    2. Synthetic: generate via SVI parameterization (offline, reproducible)

The cleaning pipeline is the same regardless of source:
    - Compute mid price from bid/ask
    - Filter by open interest and volume (liquidity)
    - Filter by moneyness bounds
    - Compute implied vol via BS inversion
    - Drop NaN / out-of-range IV values
    - Return standardized DataFrame

This is intentionally separated from the BS module to keep
data concerns away from pricing logic.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Optional

from . import config
from .black_scholes import implied_vol
from .svi_calibration import generate_svi_surface


# ════════════════════════════════════════════════════════════════════════
#  LIVE DATA (yfinance)
# ════════════════════════════════════════════════════════════════════════

def _yearfrac(t0: datetime, t1: datetime) -> float:
    """Convert timedelta to year-fraction (ACT/365.25)."""
    return (t1 - t0).total_seconds() / (365.25 * 24 * 3600)


def pull_live_chain(
    ticker: str = None,
    n_expiries: int = None,
    r: float = None,
    q: float = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Pull option chains from yfinance and compute implied vol.

    Parameters
    ----------
    ticker : underlying symbol (default: config.TICKER)
    n_expiries : how many near-term expiries to fetch (default: config.N_EXPIRIES)
    r : risk-free rate (default: config.RISK_FREE_RATE)
    q : dividend yield (default: config.DIVIDEND_YIELD)

    Returns
    -------
    df : cleaned DataFrame with columns:
         [strike, T, iv, moneyness, log_moneyness, expiry, option_type, oi, volume]
    S  : current spot price

    Raises
    ------
    ImportError : if yfinance is not installed
    RuntimeError : if data pull fails (network / market closed / etc)
    """
    if ticker is None:
        ticker = config.TICKER
    if n_expiries is None:
        n_expiries = config.N_EXPIRIES
    if r is None:
        r = config.RISK_FREE_RATE
    if q is None:
        q = config.DIVIDEND_YIELD

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for live data. Install with: pip install yfinance\n"
            "Or use --source synthetic for offline mode."
        )

    tk = yf.Ticker(ticker)

    # get spot price — use last 5 days to handle weekends
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"Failed to get price data for {ticker}. Check ticker symbol and network.")
    S = float(hist["Close"].iloc[-1])

    expiries = tk.options[:n_expiries]
    if not expiries:
        raise RuntimeError(f"No option expiries found for {ticker}.")

    now = datetime.now(timezone.utc)
    all_rows = []

    for expiry_str in expiries:
        t_exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
            hour=16, minute=0, tzinfo=timezone.utc
        )
        T = _yearfrac(now, t_exp)

        if T < config.MIN_TIME_TO_EXPIRY:
            continue

        chain = tk.option_chain(expiry_str)

        for opt_df, opt_type in [(chain.puts, "put"), (chain.calls, "call")]:
            opt = opt_df.copy()

            # mid price: more stable than last trade for illiquid strikes
            opt["mid"] = (opt["bid"] + opt["ask"]) / 2
            opt["price_used"] = np.where(opt["mid"] > 0, opt["mid"], opt["lastPrice"])

            for _, row in opt.iterrows():
                K = row["strike"]
                price = row["price_used"]
                oi = row.get("openInterest", 0) or 0
                vol = row.get("volume", 0) or 0

                # moneyness filter
                moneyness_log = np.log(K / S)
                if abs(moneyness_log) > config.MONEYNESS_BOUND:
                    continue

                # liquidity filter
                if oi < config.MIN_OPEN_INTEREST and vol < config.MIN_VOLUME:
                    continue

                # compute IV
                iv = implied_vol(price, S, K, T, r, opt_type, q)

                # IV sanity check
                if iv is None or np.isnan(iv):
                    continue
                if iv < config.MIN_IV or iv > config.MAX_IV:
                    continue

                all_rows.append({
                    "strike": K,
                    "T": T,
                    "iv": iv,
                    "moneyness": K / S,
                    "log_moneyness": moneyness_log,
                    "expiry": expiry_str,
                    "option_type": opt_type,
                    "oi": oi,
                    "volume": vol,
                })

    if not all_rows:
        raise RuntimeError(
            f"No valid IV data produced for {ticker}. "
            "Market may be closed or filters are too strict."
        )

    df = pd.DataFrame(all_rows)
    return df, S


# ════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA (SVI)
# ════════════════════════════════════════════════════════════════════════

def pull_synthetic_chain(
    S: float = 602.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Generate synthetic option chain data using SVI parameterization.

    This is the default data source — no network dependency, fully
    reproducible, and produces surfaces that look like real SPY data.

    Parameters
    ----------
    S : spot price to simulate around
    seed : random seed (default: config.SEED)

    Returns
    -------
    df : DataFrame with same schema as pull_live_chain output
    S  : spot price used
    """
    return generate_svi_surface(S=S, seed=seed)


# ════════════════════════════════════════════════════════════════════════
#  UNIFIED INTERFACE
# ════════════════════════════════════════════════════════════════════════

def get_option_data(
    source: str = "synthetic",
    ticker: str = None,
    spot: float = 602.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Main entry point for getting option data.

    Parameters
    ----------
    source : "live" or "synthetic"
    ticker : symbol for live data (ignored in synthetic mode)
    spot : spot price for synthetic data (ignored in live mode)

    Returns
    -------
    df, S : standardized option data and spot price
    """
    if source == "live":
        return pull_live_chain(ticker=ticker)
    elif source == "synthetic":
        return pull_synthetic_chain(S=spot)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'live' or 'synthetic'.")
