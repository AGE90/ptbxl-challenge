import numpy as np
import pytest

from ptbxl.preprocessing import filtering


def test_mean_removal_axis1(sample_ecg):
    out = filtering.mean_removal(sample_ecg, axis=1)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.mean(axis=1), 0, atol=1e-10)


def test_mean_removal_axis0(sample_ecg):
    # axis=0 previously produced wrong results (hardcoded transpose only
    # worked for axis=1); confirm it now generalizes correctly.
    out = filtering.mean_removal(sample_ecg, axis=0)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.mean(axis=0), 0, atol=1e-10)


def test_normalize_range_axis1(sample_ecg):
    out = filtering.normalize(sample_ecg, axis=1)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.min(axis=1), -1)
    assert np.allclose(out.max(axis=1), 1)


def test_normalize_range_axis0(sample_ecg):
    out = filtering.normalize(sample_ecg, axis=0)
    assert out.shape == sample_ecg.shape
    assert np.allclose(out.min(axis=0), -1)
    assert np.allclose(out.max(axis=0), 1)


def test_pan_tompkins_preserves_shape(sample_ecg):
    out = filtering.pan_tompkins(sample_ecg, fs=100, w=5)
    assert out.shape == sample_ecg.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("func_name", ["wander_removal", "band_pass_filtering"])
def test_odd_order_raises(sample_ecg, func_name):
    func = getattr(filtering, func_name)
    with pytest.raises(ValueError):
        func(sample_ecg, fs=100, order=3)
