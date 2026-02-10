# Methodology

## 1. Problem statement

Given a set of European option prices across multiple strikes and maturities for an equity underlying, construct a smooth implied volatility surface that captures the market's view on tail risk, skew, and term structure.

## 2. Implied volatility computation

For each observed option price $C_{mkt}$ (or $P_{mkt}$), we find the volatility $\sigma_{imp}$ that satisfies:

$$C_{BS}(S, K, T, r, \sigma_{imp}) = C_{mkt}$$

This is a root-finding problem. We use **Brent's method** (scipy.optimize.brentq) with a bracket of $[\sigma_{min}, \sigma_{max}] = [0.01\%, 500\%]$.

**Why Brent's over Newton-Raphson?** Newton-Raphson requires the vega (dC/dσ) as a derivative, and while it converges faster near the solution, it can diverge for deep OTM options where vega approaches zero. Brent's method is unconditionally convergent within the bracket — it will always find the root if one exists. For a research pipeline processing thousands of contracts, reliability is more important than per-iteration speed.

## 3. Data cleaning

Raw option data from market feeds contains significant noise:

- **Bid-ask bounce**: Using the last trade price can be misleading if the trade was at the bid or ask. We use mid-price = (bid + ask) / 2 instead.
- **Stale quotes**: Options with zero open interest or volume are likely stale. We filter by minimum OI and volume thresholds.
- **Moneyness bounds**: Very deep OTM options have tiny premiums dominated by the bid-ask spread. We restrict to |log(K/S)| < 0.25.
- **IV sanity checks**: After inversion, we drop any IV < 1% (likely data error) or > 200% (noise from illiquid quotes).

## 4. Surface interpolation

The cleaned data consists of scattered (strike, maturity, IV) triples that don't form a regular grid — different expiries have different strike spacing and coverage.

We interpolate onto a regular 100 × 60 grid using **cubic interpolation** (scipy.interpolate.griddata). Cubic interpolation produces smooth surfaces without the faceting artifacts of linear interpolation, but can produce NaN values at grid boundaries where there isn't enough data to form a complete cubic patch. We handle this with a **nearest-neighbor fallback** for any remaining NaN cells.

For noisy live data, an optional Gaussian smoothing step (scipy.ndimage.gaussian_filter) can be applied to reduce high-frequency noise while preserving the overall surface shape.

## 5. SVI parameterization (synthetic mode)

For reproducible analysis and publication-quality charts, we use a simplified SVI-inspired model:

$$\sigma_{imp}(k, T) = \sigma_{ATM}(T) + \alpha(T) \cdot k + \beta(T) \cdot k^2$$

where $k = \ln(K/S)$ is log-moneyness, and the three term-structure functions are:

- $\sigma_{ATM}(T) = a_0 + a_1 \cdot e^{-\lambda_a T}$ — ATM vol decreases with maturity
- $\alpha(T) = \alpha_0 + \alpha_1 \cdot e^{-\lambda_\alpha T}$ — skew steepens at short maturities
- $\beta(T) = \beta_0 + \beta_1 \cdot e^{-\lambda_\beta T}$ — curvature (smile) is more pronounced near-term

This is a second-order Taylor expansion of the full SVI model around ATM. It captures the three essential features of real equity index vol surfaces: level, skew, and curvature.

Parameters are calibrated to match typical SPY behavior and are stored in config.py.

## 6. Visualization

Two chart types are produced:

### 3D Surface
- X-axis: Strike (K)
- Y-axis: Time to Maturity (T) in years
- Z-axis: Implied Volatility (σ)
- Colormap: Viridis (perceptually uniform, colorblind-safe)
- Camera angle: elev=25°, azim=-55° (shows skew profile from left + term structure from front)

### 2D Skew
- X-axis: Strike (K)
- Y-axis: Implied Volatility (σ) in %
- Multiple curves for different maturities, color-coded
- ATM vertical line for reference
- Annotation highlighting the crash insurance premium

## 7. Interpretation

The key economic insights visible in the surface:

1. **Negative skew**: OTM put IV > OTM call IV at every maturity. This reflects the market's demand for downside protection — the "crash insurance premium" that Black-Scholes, with its constant-vol assumption, cannot explain.

2. **Skew steepening at short maturities**: Short-dated options show steeper skew because (a) gamma exposure is concentrated near expiry, (b) jump risk is proportionally larger relative to diffusion over short horizons, and (c) dealer hedging demand creates feedback loops.

3. **Term structure**: ATM vol typically decreases with maturity for equity indices (inverted term structure in calm markets), reflecting mean-reversion expectations. In stressed markets, the term structure can invert further or flatten.

## References

- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.
- De Marco, S. & Martini, C. (2018). Quasi-explicit calibration of Gatheral's SVI model. *Decisions in Economics and Finance*, 41(2), 285-312.
