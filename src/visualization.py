"""
Visualization module: 2D skew charts and 3D volatility surfaces.

Two backends:
    - matplotlib: high-resolution static PNGs for publications / LinkedIn
    - plotly: interactive HTML with rotation, zoom, hover tooltips

Both use a consistent dark theme to match the style commonly used
in quant finance presentations and trading floor dashboards.

Design choices:
    - Dark background (#0c0c16) reduces eye strain and looks professional
    - Viridis colormap is perceptually uniform and colorblind-safe
    - Axis labels use sigma notation (standard in vol surfaces)
    - 3D camera angle is set to show skew + term structure simultaneously
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from matplotlib import cm

import plotly.graph_objects as go

from . import config


# ════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB — 3D SURFACE (static PNG)
# ════════════════════════════════════════════════════════════════════════

def plot_surface_matplotlib(
    K_mesh: np.ndarray,
    T_mesh: np.ndarray,
    IV_mesh: np.ndarray,
    S: float,
    ticker: str = None,
    output_path: str = None,
) -> None:
    """
    Render 3D implied volatility surface as a high-res PNG.

    Parameters
    ----------
    K_mesh, T_mesh, IV_mesh : 2D arrays from surface_builder.build_surface
    S : spot price (for title annotation)
    ticker : symbol for title (default: config.TICKER)
    output_path : where to save the PNG (default: config.OUTPUT_DIR / "vol_surface_3d.png")
    """
    if ticker is None:
        ticker = config.TICKER
    if output_path is None:
        output_path = str(config.OUTPUT_DIR / "vol_surface_3d.png")

    fig = plt.figure(figsize=(config.FIG_WIDTH_3D, config.FIG_HEIGHT_3D))
    ax = fig.add_subplot(111, projection="3d")

    # plot surface — IV in percentage for readability
    surf = ax.plot_surface(
        K_mesh, T_mesh, IV_mesh * 100,
        cmap=cm.viridis,
        edgecolor="none",
        alpha=0.95,
        rstride=1,
        cstride=1,
        antialiased=True,
    )

    # labels
    ax.set_xlabel("Strike (K)", fontsize=13, labelpad=12, color="white")
    ax.set_ylabel("Time to Maturity (T)", fontsize=13, labelpad=12, color="white")
    ax.set_zlabel("Implied Volatility (\u03c3) %", fontsize=13, labelpad=12, color="white")
    ax.set_title(
        f"{ticker} \u2014 Implied Volatility Surface",
        fontsize=18, fontweight="bold", color="white", pad=20,
    )

    # dark theme styling
    ax.set_facecolor(config.DARK_BG)
    fig.patch.set_facecolor(config.DARK_BG)

    for axis in ["x", "y", "z"]:
        ax.tick_params(axis=axis, colors="white", labelsize=9)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.grid(True, alpha=0.15, color="white")

    # camera angle — chosen to show both skew and term structure
    ax.view_init(elev=config.ELEV, azim=config.AZIM)

    # colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.08)
    cbar.set_label("Implied Vol (%)", fontsize=11, color="white")
    cbar.ax.tick_params(colors="white", labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.DPI, bbox_inches="tight",
                facecolor=config.DARK_BG, edgecolor="none")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB — 2D SKEW (static PNG)
# ════════════════════════════════════════════════════════════════════════

def plot_skew_matplotlib(
    df: pd.DataFrame,
    S: float,
    ticker: str = None,
    output_path: str = None,
) -> None:
    """
    Render 2D IV skew chart with multiple maturity curves.

    Shows how skew steepens at shorter maturities — the key visual
    for the LinkedIn article.

    Parameters
    ----------
    df : DataFrame with columns [strike, T, iv]
    S : spot price (for ATM annotation)
    ticker : symbol for title
    output_path : PNG save path
    """
    if ticker is None:
        ticker = config.TICKER
    if output_path is None:
        output_path = str(config.OUTPUT_DIR / "vol_skew_2d.png")

    fig, ax = plt.subplots(figsize=(config.FIG_WIDTH_2D, config.FIG_HEIGHT_2D))
    fig.patch.set_facecolor(config.DARK_BG)
    ax.set_facecolor(config.DARK_BG)

    # find closest available maturities to our targets
    available_T = sorted(df["T"].unique())
    selected_T = []
    for target in config.SKEW_TARGET_MATURITIES:
        closest = min(available_T, key=lambda x: abs(x - target))
        if closest not in selected_T:
            selected_T.append(closest)

    for i, T_val in enumerate(selected_T):
        subset = df[df["T"] == T_val].sort_values("strike")
        label = config.MATURITY_LABELS.get(T_val, f"T={T_val:.2f}y")
        color = config.SKEW_COLORS[i % len(config.SKEW_COLORS)]
        ax.plot(subset["strike"], subset["iv"] * 100, color=color,
                linewidth=2.2, label=label)

    # ATM line
    ax.axvline(S, color="white", alpha=0.35, linestyle="--", linewidth=1)
    ylim = ax.get_ylim()
    ax.text(S + 3, ylim[1] * 0.97, f"ATM \u2248 ${S:.0f}",
            color="white", alpha=0.6, fontsize=10)

    # skew annotation
    ax.annotate(
        "\u2190 Higher IV for downside puts\n     (crash insurance premium)",
        xy=(S * 0.88, 32), fontsize=10, color="#ff6b6b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e32", edgecolor="#ff6b6b55"),
        arrowprops=dict(arrowstyle="->", color="#ff6b6b88"),
    )

    # styling
    ax.set_xlabel("Strike (K)", fontsize=13, color="white")
    ax.set_ylabel("Implied Volatility (\u03c3) %", fontsize=13, color="white")
    ax.set_title(
        f"{ticker} \u2014 Implied Volatility Skew by Maturity",
        fontsize=17, fontweight="bold", color="white",
    )
    ax.tick_params(colors="white", labelsize=10)
    ax.grid(True, alpha=0.12, color="white")

    leg = ax.legend(title="Maturity", loc="upper right", fontsize=10,
                    title_fontsize=11, facecolor="#191930", edgecolor="#ffffff30",
                    labelcolor="white")
    leg.get_title().set_color("white")

    for spine in ax.spines.values():
        spine.set_color("#333355")

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.DPI, bbox_inches="tight",
                facecolor=config.DARK_BG, edgecolor="none")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
#  PLOTLY — 3D SURFACE (interactive HTML)
# ════════════════════════════════════════════════════════════════════════

def plot_surface_plotly(
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    IV_mesh: np.ndarray,
    S: float,
    ticker: str = None,
    output_path: str = None,
) -> None:
    """
    Render interactive 3D vol surface as HTML.

    Can be opened in any browser — supports rotation, zoom, and
    hover tooltips showing exact (K, T, IV) values.
    """
    if ticker is None:
        ticker = config.TICKER
    if output_path is None:
        output_path = str(config.OUTPUT_DIR / "vol_surface_3d.html")

    fig = go.Figure(data=[go.Surface(
        x=K_grid, y=T_grid, z=IV_mesh,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(
            title=dict(text="IV (\u03c3)", font=dict(size=13, color="white")),
            thickness=18, len=0.55, tickformat=".0%",
            tickfont=dict(color="white", size=11),
        ),
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.25, roughness=0.6),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
        opacity=0.97,
        hovertemplate="Strike: %{x:.0f}<br>T: %{y:.3f}y<br>IV: %{z:.1%}<extra></extra>",
    )])

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker} \u2014 Implied Volatility Surface</b>",
            font=dict(size=22, color="white"), x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Strike (K)", font=dict(size=14, color="#ddd")),
                tickfont=dict(size=10, color="#ccc"),
                gridcolor=f"rgba(200,200,200,{config.GRID_COLOR_ALPHA})",
                backgroundcolor=config.DARK_BG,
            ),
            yaxis=dict(
                title=dict(text="Time to Maturity (T)", font=dict(size=14, color="#ddd")),
                tickfont=dict(size=10, color="#ccc"),
                gridcolor=f"rgba(200,200,200,{config.GRID_COLOR_ALPHA})",
                backgroundcolor=config.DARK_BG,
            ),
            zaxis=dict(
                title=dict(text="Implied Vol (\u03c3)", font=dict(size=14, color="#ddd")),
                tickfont=dict(size=10, color="#ccc"),
                gridcolor=f"rgba(200,200,200,{config.GRID_COLOR_ALPHA})",
                backgroundcolor=config.DARK_BG,
                tickformat=".0%",
            ),
            camera=config.PLOTLY_CAMERA,
            bgcolor=config.DARK_BG,
        ),
        paper_bgcolor=config.DARK_BG,
        font=dict(color="white"),
        width=1100, height=750,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    fig.write_html(output_path)


# ════════════════════════════════════════════════════════════════════════
#  PLOTLY — 2D SKEW (interactive HTML)
# ════════════════════════════════════════════════════════════════════════

def plot_skew_plotly(
    df: pd.DataFrame,
    S: float,
    ticker: str = None,
    output_path: str = None,
) -> None:
    """Render interactive 2D skew chart as HTML."""
    if ticker is None:
        ticker = config.TICKER
    if output_path is None:
        output_path = str(config.OUTPUT_DIR / "vol_skew_2d.html")

    available_T = sorted(df["T"].unique())
    selected_T = []
    for target in config.SKEW_TARGET_MATURITIES:
        closest = min(available_T, key=lambda x: abs(x - target))
        if closest not in selected_T:
            selected_T.append(closest)

    fig = go.Figure()
    for i, T_val in enumerate(selected_T):
        subset = df[df["T"] == T_val].sort_values("strike")
        label = config.MATURITY_LABELS.get(T_val, f"T={T_val:.2f}y")
        color = config.SKEW_COLORS[i % len(config.SKEW_COLORS)]
        fig.add_trace(go.Scatter(
            x=subset["strike"], y=subset["iv"],
            mode="lines", name=label,
            line=dict(color=color, width=2.5),
            hovertemplate="K=%{x:.0f}  IV=%{y:.1%}<extra></extra>",
        ))

    fig.add_vline(
        x=S, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text=f"ATM \u2248 ${S:.0f}",
        annotation_font=dict(color="rgba(255,255,255,0.7)", size=12),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker} \u2014 IV Skew by Maturity</b>",
            font=dict(size=20, color="white"), x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Strike (K)", font=dict(size=14, color="#ddd")),
            tickfont=dict(size=11, color="#ccc"),
            gridcolor="rgba(200,200,200,0.1)",
        ),
        yaxis=dict(
            title=dict(text="Implied Volatility (\u03c3)", font=dict(size=14, color="#ddd")),
            tickformat=".0%",
            tickfont=dict(size=11, color="#ccc"),
            gridcolor="rgba(200,200,200,0.1)",
        ),
        plot_bgcolor=config.DARK_BG,
        paper_bgcolor=config.DARK_BG,
        font=dict(color="white"),
        legend=dict(
            x=0.74, y=0.97, bgcolor="rgba(25,25,45,0.85)",
            bordercolor="rgba(255,255,255,0.15)", borderwidth=1,
            font=dict(size=12),
            title=dict(text="Maturity", font=dict(size=12, color="#ccc")),
        ),
        width=1000, height=550,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    fig.write_html(output_path)
