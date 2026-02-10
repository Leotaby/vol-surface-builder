#!/usr/bin/env python3
"""
main.py — Run the implied volatility surface pipeline.

Usage:
    python main.py                             # synthetic (default)
    python main.py --source live --ticker QQQ  # live data
"""

import argparse
import sys
import time
import numpy as np

from src.data_feed import get_option_data
from src.surface_builder import build_surface, compute_surface_statistics
from src.visualization import (
    plot_surface_matplotlib, plot_skew_matplotlib,
    plot_surface_plotly, plot_skew_plotly,
)
from src import config


def parse_args():
    p = argparse.ArgumentParser(description="Build implied volatility surfaces.")
    p.add_argument("--source", choices=["live", "synthetic"], default="synthetic")
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--spot", type=float, default=602.0)
    p.add_argument("--smooth", type=float, default=None)
    p.add_argument("--no-html", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker or config.TICKER

    print(f"\n{'='*60}")
    print(f"  Implied Volatility Surface Builder")
    print(f"  Source: {args.source}  |  Ticker: {ticker}")
    print(f"{'='*60}\n")

    # step 1: data
    t0 = time.time()
    print("[1/4] Fetching option data...")
    try:
        df, S = get_option_data(source=args.source, ticker=ticker, spot=args.spot)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    stats = compute_surface_statistics(df, S)
    print(f"       Spot: ${S:.2f}")
    print(f"       Data points: {stats['n_points']}")
    print(f"       Expiries: {stats['n_expiries']}")
    print(f"       Strike range: ${stats['strike_range'][0]:.0f} - ${stats['strike_range'][1]:.0f}")
    print(f"       IV range: {stats['iv_range'][0]:.1%} - {stats['iv_range'][1]:.1%}")
    if not np.isnan(stats.get("atm_iv_mean", np.nan)):
        print(f"       ATM IV (mean): {stats['atm_iv_mean']:.1%}")

    # step 2: build surface
    print("\n[2/4] Building interpolated surface...")
    K_grid, T_grid, K_mesh, T_mesh, IV_mesh = build_surface(df, smooth_sigma=args.smooth)
    print(f"       Grid: {config.GRID_K_POINTS} x {config.GRID_T_POINTS}")

    # step 3: static charts (matplotlib)
    print("\n[3/4] Generating static charts...")
    plot_surface_matplotlib(K_mesh, T_mesh, IV_mesh, S, ticker)
    print(f"       -> output/vol_surface_3d.png")
    plot_skew_matplotlib(df, S, ticker)
    print(f"       -> output/vol_skew_2d.png")

    # step 4: interactive HTML (plotly)
    if not args.no_html:
        print("\n[4/4] Generating interactive HTML...")
        plot_surface_plotly(K_grid, T_grid, IV_mesh, S, ticker)
        print(f"       -> output/vol_surface_3d.html")
        plot_skew_plotly(df, S, ticker)
        print(f"       -> output/vol_skew_2d.html")
    else:
        print("\n[4/4] Skipping HTML (--no-html flag)")

    # save raw data
    csv_path = config.DATA_DIR / "sample_iv_data.csv"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n       Raw data saved to data/sample_iv_data.csv")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s. Charts are in output/\n")


if __name__ == "__main__":
    main()
