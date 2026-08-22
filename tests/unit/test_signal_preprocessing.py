import numpy as np
import pytest

from ptbxl.data.signal_preprocessing import SignalPreprocessing

sp = SignalPreprocessing()


def test_mean_removal_axis1(sample_ecg):
    out = sp.mean_removal(sample_ecg, axis=1)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.mean(axis=1), 0, atol=1e-10)


def test_mean_removal_axis0(sample_ecg):
    # axis=0 previously produced wrong results (hardcoded transpose only
    # worked for axis=1); confirm it now generalizes correctly.
    out = sp.mean_removal(sample_ecg, axis=0)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.mean(axis=0), 0, atol=1e-10)


def test_normalize_range_axis1(sample_ecg):
    out = sp.normalize(sample_ecg, axis=1)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.min(axis=1), -1)
    assert np.allclose(out.max(axis=1), 1)


def test_normalize_range_axis0(sample_ecg):
    out = sp.normalize(sample_ecg, axis=0)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.min(axis=0), -1)
    assert np.allclose(out.max(axis=0), 1)


def test_pan_tompkins_preserves_shape(sample_ecg):
    out = sp.pan_tompkins(sample_ecg, fs=100, w=5)
    assert out.shape == sample_ecg.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "method_name", ["wander_removal", "band_pass_filtering"]
)
def test_odd_order_raises(sample_ecg, method_name):
    method = getattr(sp, method_name)
    with pytest.raises(ValueError):
        method(sample_ecg, fs=100, order=3)
