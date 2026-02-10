"""
Global configuration for the vol surface pipeline.

Keeps all magic numbers in one place. Override via CLI args in main.py
or by editing this file directly for persistent changes.
"""

import os
from pathlib import Path


# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# create output dir if missing (first run)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── market parameters ────────────────────────────────────────────────────
TICKER = "SPY"
RISK_FREE_RATE = 0.043          # annualized; approximate fed funds rate
DIVIDEND_YIELD = 0.013          # SPY trailing div yield — adjust per ticker


# ── data filtering ───────────────────────────────────────────────────────
MONEYNESS_BOUND = 0.25          # |log(K/S)| < 0.25 keeps ~75%-125% of spot
MIN_OPEN_INTEREST = 5           # drop strikes with OI below this
MIN_VOLUME = 5                  # drop strikes with daily volume below this
MAX_IV = 2.0                    # cap IV at 200% — anything above is noise
MIN_IV = 0.01                   # floor IV at 1%
N_EXPIRIES = 8                  # how many expiries to pull in live mode
MIN_TIME_TO_EXPIRY = 0.003      # ~1 day in year-fractions — skip anything shorter


# ── surface grid ─────────────────────────────────────────────────────────
GRID_K_POINTS = 100             # resolution along strike axis
GRID_T_POINTS = 60              # resolution along maturity axis
INTERPOLATION_METHOD = "cubic"  # "cubic", "linear", or "nearest"


# ── SVI synthetic calibration defaults ──────────────────────────────────
# these are tuned to produce realistic SPY-like surfaces
SVI_ATM_BASE = 0.18             # base ATM vol level
SVI_ATM_DECAY = 0.03            # how much ATM vol drops with maturity
SVI_ATM_LAMBDA = 1.5            # decay rate parameter
SVI_SKEW_BASE = -0.04           # long-run skew coefficient
SVI_SKEW_SHORT = -0.12          # additional skew at short maturities
SVI_SKEW_LAMBDA = 0.8           # skew decay rate
SVI_SMILE_BASE = 0.10           # long-run smile/curvature coefficient
SVI_SMILE_SHORT = 0.25          # additional curvature at short maturities
SVI_SMILE_LAMBDA = 1.0          # curvature decay rate
SVI_NOISE_STD = 0.002           # micro-noise std for realism


# ── visualization ────────────────────────────────────────────────────────
DARK_BG = "#0c0c16"
GRID_COLOR_ALPHA = 0.12
AXIS_TEXT_COLOR = "rgba(200,200,200,0.8)"
TITLE_COLOR = "white"
DPI = 200                       # matplotlib export resolution
FIG_WIDTH_3D = 14
FIG_HEIGHT_3D = 9
FIG_WIDTH_2D = 12
FIG_HEIGHT_2D = 6
COLORMAP = "viridis"

# camera angles for 3D surface (matplotlib)
ELEV = 25
AZIM = -55

# plotly camera
PLOTLY_CAMERA = dict(eye=dict(x=1.85, y=-1.55, z=0.85))

# maturity labels for legend
MATURITY_LABELS = {
    0.02: "~1 wk",
    0.04: "~2 wk",
    0.08: "~1 mo",
    0.17: "~2 mo",
    0.25: "~3 mo",
    0.50: "~6 mo",
    0.75: "~9 mo",
    1.0:  "~1 yr",
    1.5:  "~1.5 yr",
    2.0:  "~2 yr",
}

# skew line colors (mpl + plotly)
SKEW_COLORS = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#b388ff"]

# which maturities to show on the 2D skew chart
SKEW_TARGET_MATURITIES = [0.02, 0.08, 0.25, 0.50, 1.0]


# ── random seed ──────────────────────────────────────────────────────────
SEED = 42  # reproducibility for synthetic generation
