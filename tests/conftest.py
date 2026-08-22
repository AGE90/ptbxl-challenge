import numpy as np
import pytest


@pytest.fixture
def sample_ecg() -> np.ndarray:
    """Synthetic (records, samples, leads) ECG-shaped array."""
    return np.random.default_rng(0).standard_normal((3, 200, 2))
