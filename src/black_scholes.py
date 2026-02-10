"""
Black-Scholes pricing, greeks, and implied volatility inversion.

Everything here is closed-form except the IV solver, which uses
Brent's root-finding method for unconditional convergence.

References:
    Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Hull, J.C. (2018). Options, Futures, and Other Derivatives. 10th ed.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ════════════════════════════════════════════════════════════════════════
#  PRICING
# ════════════════════════════════════════════════════════════════════════

def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Compute d1 in the Black-Scholes formula.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (annualized, continuous compounding)
    sigma : volatility (annualized)
    q : continuous dividend yield (default 0)

    Returns
    -------
    float
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Compute d2 = d1 - sigma * sqrt(T)."""
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    European call price under Black-Scholes-Merton.

    Accounts for continuous dividend yield q, which makes this
    BSM rather than plain BS. For non-dividend-paying underlyings
    just leave q=0.

    Returns
    -------
    float : theoretical call price
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    European put price under Black-Scholes-Merton.

    Uses put-call parity internally rather than a separate formula,
    though the result is identical:
        P = K * e^{-rT} * N(-d2) - S * e^{-qT} * N(-d1)
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * np.exp(-q * T) * norm.cdf(-_d1)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Dispatch to call_price or put_price based on option_type."""
    if option_type.lower() in ("c", "call"):
        return call_price(S, K, T, r, sigma, q)
    elif option_type.lower() in ("p", "put"):
        return put_price(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"Unknown option_type: {option_type}. Use 'call' or 'put'.")


# ════════════════════════════════════════════════════════════════════════
#  GREEKS
# ════════════════════════════════════════════════════════════════════════

def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:
    """
    Option delta: dV/dS.

    Call delta is in [0, 1]; put delta is in [-1, 0].
    Near expiry, delta approaches a step function at the strike.
    """
    if T <= 0 or sigma <= 0:
        if option_type.lower() in ("c", "call"):
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    _d1 = d1(S, K, T, r, sigma, q)
    if option_type.lower() in ("c", "call"):
        return np.exp(-q * T) * norm.cdf(_d1)
    else:
        return np.exp(-q * T) * (norm.cdf(_d1) - 1.0)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Option gamma: d²V/dS².

    Same for calls and puts (by put-call parity). Peaks at ATM
    and increases as T → 0 — this is the "gamma risk" that makes
    short-dated option selling dangerous.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Option vega: dV/dσ.

    Returns the sensitivity per 1 unit (100%) change in vol.
    Divide by 100 to get sensitivity per 1% vol change.
    Same for calls and puts.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:
    """
    Option theta: -dV/dT (time decay per year).

    Divide by 365 for daily theta. Typically negative for long
    positions — options lose value as time passes, all else equal.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)

    # common term: time decay from gamma
    time_decay = -(S * np.exp(-q * T) * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))

    if option_type.lower() in ("c", "call"):
        return (time_decay
                + q * S * np.exp(-q * T) * norm.cdf(_d1)
                - r * K * np.exp(-r * T) * norm.cdf(_d2))
    else:
        return (time_decay
                - q * S * np.exp(-q * T) * norm.cdf(-_d1)
                + r * K * np.exp(-r * T) * norm.cdf(-_d2))


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call", q: float = 0.0) -> float:
    """
    Option rho: dV/dr.

    Sensitivity to interest rate changes. Usually small for
    short-dated options but matters for LEAPS and in rate-sensitive
    structured products.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    _d2 = d2(S, K, T, r, sigma, q)
    if option_type.lower() in ("c", "call"):
        return K * T * np.exp(-r * T) * norm.cdf(_d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-_d2)


# ════════════════════════════════════════════════════════════════════════
#  IMPLIED VOLATILITY
# ════════════════════════════════════════════════════════════════════════

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "put",
    q: float = 0.0,
    vol_lower: float = 1e-4,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """
    Compute implied volatility by inverting Black-Scholes.

    Uses Brent's method for root-finding, which is:
      - unconditionally convergent within the bracket
      - doesn't require derivatives (unlike Newton-Raphson)
      - robust to deep ITM/OTM edge cases

    The tradeoff vs Newton-Raphson: slightly slower per iteration,
    but never diverges. For a batch pipeline processing thousands
    of strikes, reliability > speed.

    Parameters
    ----------
    market_price : observed option price (ideally mid = (bid+ask)/2)
    S : spot price
    K : strike
    T : time to expiry (years)
    r : risk-free rate
    option_type : "call" or "put"
    q : dividend yield
    vol_lower : lower bracket for vol search (default 0.01%)
    vol_upper : upper bracket for vol search (default 500%)
    tol : solver tolerance

    Returns
    -------
    float : implied volatility, or NaN if solver fails

    Notes
    -----
    Common failure modes:
      - market_price <= 0 : no valid vol (option is worthless or data error)
      - market_price < intrinsic : possible early exercise premium or bad data
      - T <= 0 : expired option, IV is meaningless
      - Solver can't bracket : price is outside BS range (e.g., negative time value)
    """
    # quick sanity checks
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    # intrinsic value check — if price < intrinsic, something's wrong with the data
    # (could be bid-ask bounce, early exercise, or simply a stale quote)
    if option_type.lower() in ("c", "call"):
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < intrinsic * 0.99:
        # price below intrinsic — can't invert BS meaningfully
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(objective, vol_lower, vol_upper, xtol=tol)
    except ValueError:
        # brentq fails if f(a) and f(b) have the same sign
        # this happens when the market price is outside the BS range
        return np.nan
    except RuntimeError:
        # max iterations exceeded — extremely rare with Brent's
        return np.nan


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "put",
    q: float = 0.0,
    initial_guess: float = 0.25,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """
    Alternative IV solver using Newton-Raphson.

    Faster per iteration than Brent's (quadratic convergence near solution)
    but can diverge if initial guess is bad or vega is near zero.

    Included for comparison / educational purposes. The main pipeline
    uses Brent's method (implied_vol above) for robustness.

    Parameters
    ----------
    initial_guess : starting vol estimate (default 25%)
    max_iter : iteration limit
    tol : convergence tolerance on price difference

    Returns
    -------
    float : implied volatility or NaN
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    sigma = initial_guess

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type, q)
        v = vega(S, K, T, r, sigma, q)

        if v < 1e-12:
            # vega is essentially zero — Newton step would be huge
            # this happens deep OTM or very near expiry
            return np.nan

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / v

        # check bounds
        if sigma <= 0 or sigma > 10.0:
            return np.nan

    # didn't converge
    return np.nan
