import neurokit2 as nk
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


def test_wavelet_features_shapes_and_values():
    x = np.random.default_rng(3).standard_normal((3, 200, 2))

    features = bf.wavelet_features(x, wavelet='db4', level=4, axis=1)

    expected_keys = {'nrg', 'mean', 'std', 'skew', 'kurt', 'entropy', 'dom'}
    assert len(features) == 5  # level=4 -> 1 approximation + 4 detail dicts
    for i, level_features in enumerate(features, start=1):
        assert set(level_features) == {f'l{i}{k}' for k in expected_keys}
        for value in level_features.values():
            assert value.shape == (3, 2)
            assert np.all(np.isfinite(value))
        assert np.all(level_features[f'l{i}entropy'] >= 0)
        assert np.all(level_features[f'l{i}dom'] >= 0)


def test_time_domain_features_shapes_and_values():
    x = np.random.default_rng(2).standard_normal((3, 200, 2))

    features = bf.time_domain_features(x, axis=1)

    expected_keys = {'std', 'rms', 'ptp', 'skew', 'kurtosis', 'zcr'}
    assert set(features) == expected_keys
    for value in features.values():
        assert value.shape == (3, 2)
        assert np.all(np.isfinite(value))

    np.testing.assert_allclose(features['std'], np.std(x, axis=1))
    assert np.all(features['rms'] >= 0)
    assert np.all(features['zcr'] >= 0) and np.all(features['zcr'] <= 1)


def test_heart_rate_features_detects_known_rate():
    # Two records, evenly spaced unit pulses every 60 samples at fs=100 Hz
    # -> RR = 0.6s -> HR = 100 bpm, zero RR variability.
    fs = 100
    n_samples = 500
    peak_spacing = 60
    x = np.zeros((2, n_samples))
    x[:, ::peak_spacing] = 1.0

    features = bf.heart_rate_features(x, fs=fs)

    assert np.allclose(features['heart_rate'], 100, atol=1)
    assert np.allclose(features['rr_cv'], 0, atol=1e-6)
    assert np.all(features['n_peaks'] == n_samples // peak_spacing)


def test_wave_spectral_features_returns_finite_values_for_clean_signal():
    fs = 100
    sig = nk.ecg_simulate(duration=8, sampling_rate=fs, heart_rate=75, random_state=42)
    x = np.repeat(sig[None, :, None], 12, axis=2)  # (1, n, 12): same trace on all leads
    x = np.repeat(x, 2, axis=0)  # (2, n, 12): two identical records

    features = bf.wave_spectral_features(x, fs=fs, lead_idx=0)

    expected_keys = {
        'n_beats',
        'P_dominant_freq', 'P_spectral_entropy',
        'QRS_dominant_freq', 'QRS_spectral_entropy',
        'T_dominant_freq', 'T_spectral_entropy',
    }
    assert set(features) == expected_keys
    assert np.all(features['n_beats'] >= 5)  # ~75bpm over 8s -> ~10 beats expected
    for key in expected_keys - {'n_beats'}:
        assert features[key].shape == (2, 12)
        assert np.all(np.isfinite(features[key]))


def test_wave_spectral_features_nan_fills_on_no_beats_detected():
    fs = 100
    x = np.zeros((1, 800, 12))  # flat line: no beats, delineation finds nothing

    features = bf.wave_spectral_features(x, fs=fs, lead_idx=0)

    assert features['n_beats'][0] == 0
    assert np.all(np.isnan(features['P_dominant_freq'][0]))
    assert np.all(np.isnan(features['QRS_spectral_entropy'][0]))
