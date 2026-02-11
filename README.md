# Implied Volatility Surface Builder

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

End-to-end pipeline for constructing, calibrating, and visualizing implied volatility surfaces from equity option chains. Built for research, risk analysis, and portfolio-level vol monitoring.

<p align="center">
  <img src="output/vol_surface_3d.png" width="720" alt="SPY implied volatility surface">
</p>

---

## Motivation

The Black-Scholes model assumes constant volatility a convenient fiction that breaks immediately when you look at real option prices. Implied volatility varies across strikes (skew) and maturities (term structure), forming a surface that encodes the market's view on tail risk, jump risk, and hedging costs.

This project builds that surface from scratch:

1. Pull live option chains (or use synthetic calibration data)
2. Compute implied volatility by numerically inverting Black-Scholes
3. Clean and filter the raw IV grid (bid-ask noise, stale quotes, illiquid strikes)
4. Interpolate onto a smooth surface via cubic splines
5. Visualize in 2D (skew by maturity) and 3D (full surface)

The goal is **not** just pretty charts it's a working analytical tool that reveals how the market prices crash insurance, how skew steepens in stress, and where Black-Scholes breaks down.

---

## What the skew actually tells you

<p align="center">
  <img src="output/vol_skew_2d.png" width="720" alt="IV skew across maturities">
</p>

The 2D chart above shows implied volatility by strike for different maturities. A few things jump out:

- **Negative skew is persistent**: lower strikes (OTM puts) trade at higher IV than upper strikes (OTM calls). This is the market pricing crash insurance the premium for unhedgeable jump risk that Black-Scholes assumes away.
- **Short-dated skew is steeper**: the ~1 week curve has the most dramatic slope. Gamma exposure is concentrated, and dealers need wider spreads to compensate.
- **Skew flattens with maturity**: the ~1 year curve is much smoother. Over longer horizons, the law of large numbers partially absorbs individual jump events.

This isn't academic skew dynamics directly affect delta-hedging P&L, structured product pricing, and tail risk measurement in any portfolio context.

---

## Mathematical formulation

### 1. Black-Scholes-Merton pricing

Assume the underlying follows geometric Brownian motion. Let $S_0$ be spot, $K$ strike, $T$ time to maturity (years), $r$ risk-free rate, $q$ continuous dividend yield, and $\sigma$ volatility. Define:

$$d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

The value of a European call is:

$$C(S_0, K, T) = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

The corresponding European put is:

$$P(S_0, K, T) = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

where $N(\cdot)$ is the standard normal CDF. These are implemented in [`src/black_scholes.py`](src/black_scholes.py).

### 2. Implied volatility as an inversion problem

Given a market option price $V_{mkt}$ (mid-price after bid-ask and liquidity filtering), implied volatility $\sigma_{imp}$ is defined as the root of:

$$f(\sigma) = V_{BS}(S_0, K, T, r, q, \sigma) - V_{mkt} = 0$$

There is no closed-form inverse for $\sigma$, so we solve numerically. This project uses **Brent's method**  a hybrid of bisection, secant, and inverse quadratic interpolation that is unconditionally convergent within a bracket $[\sigma_L, \sigma_U]$:

$$\sigma^* : f(\sigma) = 0, \quad \sigma \in [\sigma_L, \sigma_U]$$

A Newton-Raphson solver (using vega $\partial V / \partial \sigma$ as the derivative) is also included for comparison but is not used in the main pipeline due to divergence risk on deep OTM options.

### 3. SVI parameterization (Gatheral, 2004)

For each expiry slice, define log-moneyness $k = \ln(K/F)$ where $F$ is the forward price. The **Stochastic Volatility Inspired (SVI)** model parameterizes total implied variance $w(k) = \sigma_{imp}^2(k) \cdot T$ as:

$$w(k) = a + b\left(\rho\,(k - m) + \sqrt{(k-m)^2 + \sigma^2}\right)$$

with parameters $a \in \mathbb{R}$, $b > 0$, $\rho \in (-1,1)$, $m \in \mathbb{R}$, $\sigma > 0$.

Implied volatility is then recovered as:

$$\sigma_{imp}(k) = \sqrt{\frac{w(k)}{T}}$$

**Calibration** fits these five parameters per expiry slice by minimizing:

$$\min_{\theta} \sum_{n=1}^{N} (w_{\theta}(k_n) - w_{mkt}(k_n))^2 \quad s.t. \; b > 0, \; \sigma > 0, \; |\rho| < 1$$

where $w_{mkt}(k_n) = (\sigma_n^{mkt})^2 T$. Solved via L-BFGS-B in [`src/svi_calibration.py`](src/svi_calibration.py).

### 4. From scattered quotes to the 3D surface

After computing $\sigma_{imp}(K_i, T_j)$ on an irregular grid (strikes differ by expiry), the smooth 3D surface is constructed by interpolating the scattered observations:

Given $N$ observed triples $\{(K_n, T_n, \sigma_n)\}_{n=1}^{N}$, define a dense regular evaluation grid $(K, T) \in \mathcal{G}$. The surface estimate is:

$$\hat{\sigma}(K, T) = \mathcal{I}(\{(K_n, T_n, \sigma_n)\})$$

where $\mathcal{I}(\cdot)$ is a scattered-data interpolation operator specifically, **cubic interpolation** inside the convex hull of the data and **nearest-neighbor fallback** at grid boundaries. In code, this corresponds to `scipy.interpolate.griddata` with `method="cubic"`.

This $\hat{\sigma}(K, T)$ is exactly what is rendered as the 3D implied volatility surface.

---

## Project structure

```
vol-surface-builder/
├── src/
│   ├── __init__.py
│   ├── black_scholes.py          # BS pricing, greeks, IV inversion
│   ├── data_feed.py              # option chain retrieval + cleaning
│   ├── surface_builder.py        # interpolation + surface construction
│   ├── visualization.py          # matplotlib + plotly charting
│   ├── svi_calibration.py        # SVI parametric fit (Gatheral)
│   └── config.py                 # constants, defaults, paths
├── notebooks/
│   └── vol_surface_analysis.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_black_scholes.py
│   ├── test_surface_builder.py
│   └── test_svi.py
├── data/
│   └── sample_iv_data.csv
├── output/
│   ├── vol_surface_3d.png
│   ├── vol_skew_2d.png
│   ├── vol_surface_3d.html
│   └── vol_skew_2d.html
├── docs/
│   └── methodology.md
├── main.py
├── requirements.txt
├── setup.py
├── .gitignore
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/Leotaby/vol-surface-builder.git
cd vol-surface-builder
pip install -r requirements.txt
python main.py
```

Output charts land in `output/`. Interactive HTML versions can be opened in any browser.

To use live market data instead of synthetic calibration:

```bash
python main.py --source live --ticker SPY
```

> **Note:** live data requires `yfinance` and an internet connection. The default mode uses SVI-calibrated synthetic data which runs offline and produces publication-quality results.

---

## Key modules

### `black_scholes.py`

Core pricing engine. Implements closed-form BS for European calls/puts, first-order greeks (delta, gamma, vega, theta, rho), and implied volatility inversion via Brent's method with robust bracketing. The IV solver handles edge cases: deep ITM/OTM, near-expiry, and zero-bid situations that cause naive solvers to diverge.

### `data_feed.py`

Handles option chain retrieval from `yfinance` (equities) with fallback to synthetic generation via SVI parameterization. Applies a cleaning pipeline: mid-price calculation, liquidity filters (OI + volume thresholds), moneyness bounds, and stale-quote detection. Returns a standardized DataFrame regardless of source.

### `surface_builder.py`

Takes cleaned (strike, maturity, IV) triples and constructs a smooth surface. Uses `scipy.interpolate.griddata` with cubic interpolation, falling back to nearest-neighbor at domain boundaries. Handles the messy reality of irregular option grids strikes don't align across expiries, and some maturities have sparse coverage.

### `svi_calibration.py`

Implements the Stochastic Volatility Inspired (SVI) parameterization from Gatheral (2004). The SVI model fits total implied variance as a function of log-moneyness with five parameters: `a` (overall variance level), `b` (slope), `rho` (rotation/skew), `m` (translation), `sigma` (curvature). This project uses a simplified version for synthetic data generation that captures the essential features: negative skew, smile curvature at wings, and term structure flattening.

### `visualization.py`

Charting module with two backends: `matplotlib` for high-resolution static PNGs (dark theme, publication-ready) and `plotly` for interactive HTML with rotation/zoom. Both produce consistent styling dark background, viridis colormap, properly labeled axes.

---

## Configuration

Edit `src/config.py` or pass CLI arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKER` | `"SPY"` | Underlying symbol |
| `RISK_FREE_RATE` | `0.043` | Annualized risk-free rate |
| `MONEYNESS_BOUND` | `0.25` | Max abs(log-moneyness) to include |
| `MIN_OI` | `5` | Minimum open interest filter |
| `MIN_VOLUME` | `5` | Minimum daily volume filter |
| `N_EXPIRIES` | `8` | Number of expiries to pull (live mode) |
| `GRID_K_POINTS` | `100` | Strike-axis grid resolution |
| `GRID_T_POINTS` | `60` | Maturity-axis grid resolution |

---

## Technical notes

**Why Brent's method for IV?** Newton-Raphson is faster in theory but requires a good initial guess and can diverge for deep OTM options where vega is near zero. Brent's method is unconditionally convergent within a bracket slower per iteration, but never fails. For a research tool where reliability matters more than microseconds, this is the right tradeoff.

**Why SVI for synthetic data?** The SVI parameterization is arbitrage-free (under parameter constraints) and captures the three main features of real vol surfaces: level, skew, and curvature. It's used extensively on equity index desks. The alternative pulling live data introduces noise, market-hours dependency, and reproducibility issues. SVI gives clean, realistic surfaces that are identical every time you run the code.

**Why not fit a full stochastic vol model?** Heston, SABR, and jump-diffusion models are more realistic but also more complex to calibrate. This project focuses on the **construction and visualization** of the vol surface, not on exotic model calibration. The `svi_calibration.py` module is intentionally kept simple so the pipeline is transparent. Extending to Heston or local vol is straightforward from this foundation.

---

## Extending this

Some natural next steps if you want to take this further:

- **SVI arbitrage constraints**: enforce no-butterfly and no-calendar spread arbitrage in the fitted surface (Gatheral & Jacquier, 2014)
- **Heston calibration**: fit the Heston stochastic vol model to the surface via characteristic function pricing + Levenberg-Marquardt
- **Greeks surface**: compute and plot delta, gamma, vega surfaces useful for book-level risk decomposition
- **Real-time streaming**: replace batch pulls with websocket feeds from IBKR or Polygon for live surface updates
- **Skew analytics**: track 25-delta risk reversal and butterfly over time as regime indicators

---

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy, 81*(3), 637-654. https://doi.org/10.1086/260062

- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics, 3*(1–2), 125-144. https://doi.org/10.1016/0304-405X(76)90022-2

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Global Derivatives & Risk Management (GDRM), Madrid* (presentation / technical note).

- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance, 14*(1), 59-71. https://doi.org/10.1080/14697688.2013.819986

- Dupire, B. (1994). Pricing with a Smile. *Risk Magazine*, January 1994, 18-20.

---

## License

MIT - see [LICENSE](LICENSE).
