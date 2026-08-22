import numpy as np

from ptbxl.features.build_features import BuildFeatures

bf = BuildFeatures()


def test_spectral_entropy_no_nan_on_zero_power_row():
    # Column 0 is all-zero (no power at all); column 1 has real power.
    # The zero-guard must be per-axis, not a single global sum, or column 0
    # NaNs out even though the (wrong) global guard reports "safe to divide".
    x = np.zeros((200, 2))
    x[:, 1] = np.random.default_rng(1).standard_normal(200)

    entropy = bf.spectral_entropy(x, fs=100)

    assert np.all(np.isfinite(entropy))
    assert entropy[0] == 0
