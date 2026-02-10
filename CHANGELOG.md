# Changelog

## [0.2.1] — 2025-02-09

### Added
- SVI calibration module with L-BFGS-B parameter fitting
- Newton-Raphson IV solver (alongside existing Brent's method)
- Surface statistics: ATM IV, 25Δ skew proxy
- Gaussian smoothing option for noisy live data
- GitHub Actions CI workflow for Python 3.10/3.11/3.12
- Exploratory analysis notebook

### Changed
- Refactored visualization into separate matplotlib/plotly functions
- Improved data cleaning pipeline: moneyness filters + liquidity thresholds
- Config module now centralizes all constants

### Fixed
- Edge case in IV solver where deep OTM puts with near-zero vega caused divergence
- NaN handling at surface grid boundaries (cubic → nearest-neighbor fallback)

## [0.1.0] — 2025-01-22

### Added
- Initial release
- Black-Scholes pricing engine with greeks (delta, gamma, vega, theta, rho)
- Implied volatility inversion via Brent's method
- Option chain retrieval from yfinance
- Cubic surface interpolation
- 3D surface and 2D skew visualization (matplotlib + plotly)
- SVI-based synthetic data generation
- Basic test suite
