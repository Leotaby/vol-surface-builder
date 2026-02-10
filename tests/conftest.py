"""
Shared test fixtures and pytest configuration.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensure test reproducibility."""
    np.random.seed(42)
    yield
