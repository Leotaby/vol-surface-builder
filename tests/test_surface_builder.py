"""
Tests for surface construction and interpolation.
"""

import pytest
import numpy as np
import pandas as pd
from src.surface_builder import build_surface, extract_skew_slices, compute_surface_statistics
from src.svi_calibration import generate_svi_surface


@pytest.fixture
def sample_data():
    """Generate a small synthetic dataset for testing."""
    np.random.seed(42)
    df, S = generate_svi_surface(S=600.0, seed=42)
    return df, S


class TestBuildSurface:

    def test_output_shapes(self, sample_data):
        df, S = sample_data
        K_grid, T_grid, K_mesh, T_mesh, IV_mesh = build_surface(df, n_k=50, n_t=30)
        assert K_grid.shape == (50,)
        assert T_grid.shape == (30,)
        assert K_mesh.shape == (30, 50)
        assert T_mesh.shape == (30, 50)
        assert IV_mesh.shape == (30, 50)

    def test_no_nans_in_output(self, sample_data):
        """After nearest-neighbor fallback, there should be no NaNs."""
        df, S = sample_data
        _, _, _, _, IV_mesh = build_surface(df)
        assert not np.any(np.isnan(IV_mesh))

    def test_iv_values_reasonable(self, sample_data):
        """Interpolated IV should be within [0, 1] for SPY-like data."""
        df, S = sample_data
        _, _, _, _, IV_mesh = build_surface(df)
        assert IV_mesh.min() > 0.0
        assert IV_mesh.max() < 1.0

    def test_grid_bounds(self, sample_data):
        """Grid should span most of the input data range."""
        df, S = sample_data
        K_grid, T_grid, _, _, _ = build_surface(df)
        assert K_grid.min() <= df["strike"].quantile(0.05)
        assert K_grid.max() >= df["strike"].quantile(0.95)
        assert T_grid.min() <= df["T"].min() + 0.01
        assert T_grid.max() >= df["T"].max() - 0.01

    def test_smoothing(self, sample_data):
        """Smoothing should reduce surface roughness."""
        df, S = sample_data
        _, _, _, _, IV_raw = build_surface(df, smooth_sigma=None)
        _, _, _, _, IV_smooth = build_surface(df, smooth_sigma=2.0)
        # smoothed surface should have smaller max gradient
        grad_raw = np.max(np.abs(np.diff(IV_raw, axis=0)))
        grad_smooth = np.max(np.abs(np.diff(IV_smooth, axis=0)))
        assert grad_smooth <= grad_raw


class TestExtractSlices:

    def test_correct_number_of_slices(self, sample_data):
        df, S = sample_data
        slices = extract_skew_slices(df, target_maturities=[0.02, 0.25, 1.0])
        assert len(slices) >= 1  # at least one match
        assert len(slices) <= 3  # at most 3 unique

    def test_slices_sorted_by_strike(self, sample_data):
        df, S = sample_data
        slices = extract_skew_slices(df)
        for T_val, subset in slices.items():
            assert list(subset["strike"]) == sorted(subset["strike"])


class TestStatistics:

    def test_stats_keys(self, sample_data):
        df, S = sample_data
        stats = compute_surface_statistics(df, S)
        assert "n_points" in stats
        assert "n_expiries" in stats
        assert "strike_range" in stats
        assert "iv_range" in stats

    def test_stats_values(self, sample_data):
        df, S = sample_data
        stats = compute_surface_statistics(df, S)
        assert stats["n_points"] == len(df)
        assert stats["n_expiries"] == df["T"].nunique()
        assert stats["iv_range"][0] < stats["iv_range"][1]
