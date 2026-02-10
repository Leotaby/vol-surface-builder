"""
Tests for SVI parameterization and calibration.
"""

import pytest
import numpy as np
from src.svi_calibration import (
    svi_total_variance,
    svi_implied_vol,
    generate_svi_surface,
    calibrate_svi_slice,
)


class TestSVIEvaluation:

    def test_total_variance_positive_atm(self):
        """Total variance at ATM (k=0) should be positive for valid params."""
        k = np.array([0.0])
        w = svi_total_variance(k, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        assert w[0] > 0

    def test_total_variance_symmetric_wings(self):
        """With rho=0, surface should be symmetric around m."""
        k = np.linspace(-0.3, 0.3, 100)
        w = svi_total_variance(k, a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        # check symmetry: w(k) == w(-k) when rho=0 and m=0
        w_flip = w[::-1]
        np.testing.assert_allclose(w, w_flip, atol=1e-12)

    def test_negative_rho_creates_skew(self):
        """Negative rho should make left wing higher than right (skew)."""
        k = np.array([-0.2, 0.0, 0.2])
        w = svi_total_variance(k, a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
        assert w[0] > w[1]  # left wing > ATM
        assert w[1] < w[2] or w[0] > w[2]  # skew visible

    def test_implied_vol_from_total_variance(self):
        """IV = sqrt(w/T) should be consistent."""
        k = np.array([0.0])
        T = 0.25
        iv = svi_implied_vol(k, T, a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        w = svi_total_variance(k, a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        expected = np.sqrt(w / T)
        np.testing.assert_allclose(iv, expected, atol=1e-12)


class TestSVIGeneration:

    def test_generates_dataframe(self):
        df, S = generate_svi_surface(S=600.0, seed=42)
        assert len(df) > 100
        assert "strike" in df.columns
        assert "T" in df.columns
        assert "iv" in df.columns

    def test_iv_range_realistic(self):
        df, S = generate_svi_surface(S=600.0, seed=42)
        assert df["iv"].min() > 0.01
        assert df["iv"].max() < 1.0

    def test_negative_skew_present(self):
        """Lower strikes should have higher IV (negative skew)."""
        df, S = generate_svi_surface(S=600.0, seed=42)
        # check for the shortest maturity
        short_T = df["T"].min()
        short = df[df["T"] == short_T].sort_values("strike")
        # first quarter should have higher IV than last quarter on average
        n = len(short) // 4
        low_strike_iv = short.head(n)["iv"].mean()
        high_strike_iv = short.tail(n)["iv"].mean()
        assert low_strike_iv > high_strike_iv, "Negative skew not present"

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        df1, _ = generate_svi_surface(S=600.0, seed=123)
        df2, _ = generate_svi_surface(S=600.0, seed=123)
        np.testing.assert_array_equal(df1["iv"].values, df2["iv"].values)

    def test_different_seeds_differ(self):
        """Different seeds should produce different noise."""
        df1, _ = generate_svi_surface(S=600.0, seed=1)
        df2, _ = generate_svi_surface(S=600.0, seed=2)
        assert not np.allclose(df1["iv"].values, df2["iv"].values)


class TestSVICalibration:

    def test_calibrate_atm_slice(self):
        """Fit SVI to generated data and check RMSE is small."""
        df, S = generate_svi_surface(S=600.0, seed=42)
        T_val = 0.25
        slice_df = df[df["T"] == T_val]
        k = slice_df["log_moneyness"].values
        iv = slice_df["iv"].values

        result = calibrate_svi_slice(k, iv, T_val)
        assert result["converged"]
        assert result["rmse"] < 0.01  # RMSE < 1% vol

    def test_calibrated_rho_negative(self):
        """Fitted rho should be negative for skewed data."""
        df, S = generate_svi_surface(S=600.0, seed=42)
        T_val = 0.08  # short maturity with steep skew
        slice_df = df[df["T"] == T_val]
        k = slice_df["log_moneyness"].values
        iv = slice_df["iv"].values

        result = calibrate_svi_slice(k, iv, T_val)
        assert result["rho"] < 0, "Rho should be negative for equity skew"
