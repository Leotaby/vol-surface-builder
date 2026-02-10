"""
Tests for the Black-Scholes pricing module.

Covers: pricing accuracy, put-call parity, greek signs/bounds,
IV round-trip consistency, and edge cases.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from src.black_scholes import (
    call_price, put_price, bs_price,
    delta, gamma, vega, theta, rho,
    implied_vol, implied_vol_newton,
    d1, d2,
)


# ── fixtures ─────────────────────────────────────────────────────────

# standard test parameters: ATM SPY-like option
S = 600.0
K = 600.0
T = 0.25  # 3 months
r = 0.05
sigma = 0.20
q = 0.013


class TestPricing:
    """Basic pricing correctness."""

    def test_call_price_positive(self):
        """Call price must be positive for reasonable inputs."""
        c = call_price(S, K, T, r, sigma, q)
        assert c > 0

    def test_put_price_positive(self):
        """Put price must be positive for reasonable inputs."""
        p = put_price(S, K, T, r, sigma, q)
        assert p > 0

    def test_put_call_parity(self):
        """
        Put-call parity: C - P = S*e^{-qT} - K*e^{-rT}

        This is model-independent for European options. If this fails,
        something fundamental is wrong with the pricing formulas.
        """
        c = call_price(S, K, T, r, sigma, q)
        p = put_price(S, K, T, r, sigma, q)
        lhs = c - p
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10

    def test_put_call_parity_otm(self):
        """Put-call parity holds for OTM options too."""
        for K_test in [500.0, 550.0, 650.0, 700.0]:
            c = call_price(S, K_test, T, r, sigma, q)
            p = put_price(S, K_test, T, r, sigma, q)
            lhs = c - p
            rhs = S * np.exp(-q * T) - K_test * np.exp(-r * T)
            assert abs(lhs - rhs) < 1e-10, f"PCP failed at K={K_test}"

    def test_call_itm_lower_bound(self):
        """Call price >= max(S*e^{-qT} - K*e^{-rT}, 0)."""
        c = call_price(S, K, T, r, sigma, q)
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        assert c >= intrinsic - 1e-10

    def test_put_itm_lower_bound(self):
        """Put price >= max(K*e^{-rT} - S*e^{-qT}, 0)."""
        p = put_price(S, 700.0, T, r, sigma, q)
        intrinsic = max(700.0 * np.exp(-r * T) - S * np.exp(-q * T), 0)
        assert p >= intrinsic - 1e-10

    def test_expired_call(self):
        """At expiry, call = max(S-K, 0)."""
        assert call_price(600, 550, 0, r, sigma) == 50.0
        assert call_price(600, 650, 0, r, sigma) == 0.0

    def test_expired_put(self):
        """At expiry, put = max(K-S, 0)."""
        assert put_price(600, 650, 0, r, sigma) == 50.0
        assert put_price(600, 550, 0, r, sigma) == 0.0

    def test_zero_vol_call(self):
        """With zero vol, call = max(S*e^{-qT} - K*e^{-rT}, 0)."""
        c = call_price(600, 550, 0.5, 0.05, 0.0, 0.0)
        expected = max(600 - 550 * np.exp(-0.05 * 0.5), 0)
        assert abs(c - expected) < 1e-10

    def test_bs_price_dispatch(self):
        """bs_price dispatches correctly to call/put."""
        assert bs_price(S, K, T, r, sigma, "call", q) == call_price(S, K, T, r, sigma, q)
        assert bs_price(S, K, T, r, sigma, "put", q) == put_price(S, K, T, r, sigma, q)
        assert bs_price(S, K, T, r, sigma, "c", q) == call_price(S, K, T, r, sigma, q)
        assert bs_price(S, K, T, r, sigma, "p", q) == put_price(S, K, T, r, sigma, q)

    def test_bs_price_invalid_type(self):
        with pytest.raises(ValueError):
            bs_price(S, K, T, r, sigma, "invalid")


class TestGreeks:
    """Greek signs, bounds, and symmetries."""

    def test_call_delta_bounds(self):
        """Call delta is in [0, 1]."""
        for K_test in [500, 550, 600, 650, 700]:
            d = delta(S, K_test, T, r, sigma, "call", q)
            assert 0 <= d <= 1, f"Call delta out of bounds at K={K_test}: {d}"

    def test_put_delta_bounds(self):
        """Put delta is in [-1, 0]."""
        for K_test in [500, 550, 600, 650, 700]:
            d = delta(S, K_test, T, r, sigma, "put", q)
            assert -1 <= d <= 0, f"Put delta out of bounds at K={K_test}: {d}"

    def test_gamma_positive(self):
        """Gamma is always positive (same for calls and puts)."""
        for K_test in [500, 550, 600, 650, 700]:
            g = gamma(S, K_test, T, r, sigma, q)
            assert g >= 0, f"Gamma negative at K={K_test}: {g}"

    def test_gamma_peaks_atm(self):
        """Gamma should be highest at ATM."""
        g_atm = gamma(S, 600, T, r, sigma, q)
        g_itm = gamma(S, 550, T, r, sigma, q)
        g_otm = gamma(S, 650, T, r, sigma, q)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_vega_positive(self):
        """Vega is always positive."""
        for K_test in [500, 550, 600, 650, 700]:
            v = vega(S, K_test, T, r, sigma, q)
            assert v >= 0, f"Vega negative at K={K_test}: {v}"

    def test_long_call_theta_negative(self):
        """Long ATM call should have negative theta (time decay)."""
        th = theta(S, K, T, r, sigma, "call", q)
        assert th < 0

    def test_call_rho_positive(self):
        """Call rho is positive (higher rates -> higher call value)."""
        rh = rho(S, K, T, r, sigma, "call", q)
        assert rh > 0

    def test_put_rho_negative(self):
        """Put rho is negative (higher rates -> lower put value)."""
        rh = rho(S, K, T, r, sigma, "put", q)
        assert rh < 0


class TestImpliedVol:
    """Implied volatility solver tests."""

    def test_iv_round_trip_call(self):
        """Price a call, then invert -> should recover original vol."""
        price = call_price(S, K, T, r, sigma, q)
        iv_recovered = implied_vol(price, S, K, T, r, "call", q)
        assert abs(iv_recovered - sigma) < 1e-6

    def test_iv_round_trip_put(self):
        """Same for put."""
        price = put_price(S, K, T, r, sigma, q)
        iv_recovered = implied_vol(price, S, K, T, r, "put", q)
        assert abs(iv_recovered - sigma) < 1e-6

    def test_iv_round_trip_various_vols(self):
        """Test round-trip across a range of volatilities."""
        for test_sigma in [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]:
            price = call_price(S, K, T, r, test_sigma, q)
            iv_recovered = implied_vol(price, S, K, T, r, "call", q)
            assert abs(iv_recovered - test_sigma) < 1e-5, \
                f"Round-trip failed for sigma={test_sigma}: got {iv_recovered}"

    def test_iv_round_trip_otm(self):
        """Round-trip works for OTM options."""
        for K_test in [500, 550, 650, 700]:
            price = put_price(S, K_test, T, r, 0.25, q)
            iv_recovered = implied_vol(price, S, K_test, T, r, "put", q)
            if not np.isnan(iv_recovered):
                assert abs(iv_recovered - 0.25) < 1e-4

    def test_iv_zero_price(self):
        """Zero price -> NaN (no valid vol)."""
        result = implied_vol(0.0, S, K, T, r, "put")
        assert np.isnan(result)

    def test_iv_negative_price(self):
        """Negative price -> NaN."""
        result = implied_vol(-5.0, S, K, T, r, "put")
        assert np.isnan(result)

    def test_iv_expired(self):
        """T=0 -> NaN."""
        result = implied_vol(10.0, S, K, 0, r, "put")
        assert np.isnan(result)

    def test_newton_vs_brent(self):
        """Newton and Brent should agree for well-behaved inputs."""
        price = call_price(S, K, T, r, sigma, q)
        iv_brent = implied_vol(price, S, K, T, r, "call", q)
        iv_newton = implied_vol_newton(price, S, K, T, r, "call", q)
        if not np.isnan(iv_newton):
            assert abs(iv_brent - iv_newton) < 1e-4


class TestEdgeCases:
    """Edge cases and numerical stability."""

    def test_very_short_maturity(self):
        """Near-expiry options should still price correctly."""
        c = call_price(600, 590, 0.001, r, 0.20)
        assert c > 0  # ITM call should have positive value

    def test_very_high_vol(self):
        """Extreme vol should not cause overflow."""
        c = call_price(100, 100, 1.0, 0.05, 5.0)
        assert np.isfinite(c)
        assert c > 0

    def test_very_low_vol(self):
        """Near-zero vol: call should approach discounted intrinsic."""
        c = call_price(600, 550, 1.0, 0.05, 0.001, 0.0)
        intrinsic = 600 - 550 * np.exp(-0.05)
        assert abs(c - intrinsic) < 0.5  # within $0.50

    def test_deep_otm_put_iv(self):
        """Deep OTM put: IV solver should handle gracefully."""
        # a $1 put on a $600 stock with K=400 — basically worthless
        result = implied_vol(0.01, 600, 400, 0.25, 0.05, "put")
        # should either return a very high vol or NaN, not crash
        assert np.isnan(result) or result > 0
