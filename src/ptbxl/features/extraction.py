"""Composed feature extraction, config-driven (see `ptbxl.config.FeaturesConfig`).

Column naming matches the `<StatPrefix>_<LeadCode>` convention used throughout
notebooks 03/04 (e.g. `DF_II`, `l3nrg_V1`, `Std_V6`).
"""

import numpy as np
import pandas as pd

from ptbxl.config import FeaturesConfig
from ptbxl.data.ingestion import LEADS
from ptbxl.features.spectral import (
    dominant_frequency,
    spectral_entropy,
    wave_spectral_features,
)
from ptbxl.features.temporal import heart_rate_features, time_domain_features
from ptbxl.features.wavelet import wavelet_features

_TIME_DOMAIN_PREFIXES = {
    "std": "Std",
    "rms": "RMS",
    "ptp": "PTP",
    "skew": "Skew",
    "kurtosis": "Kurt",
    "zcr": "ZCR",
}
_WAVE_SPECTRAL_PREFIXES = {
    "P_dominant_freq": "P_DF",
    "P_spectral_entropy": "P_SpecEn",
    "QRS_dominant_freq": "QRS_DF",
    "QRS_spectral_entropy": "QRS_SpecEn",
    "T_dominant_freq": "T_DF",
    "T_spectral_entropy": "T_SpecEn",
}


def _per_lead_frame(values: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(values, columns=[f"{prefix}_{lead}" for lead in LEADS])


def extract_features(
    signal: np.ndarray,
    fs: float,
    config: FeaturesConfig,
    qrs_envelope: np.ndarray | None = None,
    lead_idx: int = 1,
) -> pd.DataFrame:
    """Builds a feature DataFrame from a preprocessed `(records, samples, leads)` array.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal (e.g. `preprocessing.pipeline.preprocess_ecg` output).
    fs : float
        Sampling frequency of the signal in Hz.
    config : FeaturesConfig
        Toggles which feature groups to compute.
    qrs_envelope : np.ndarray, optional
        Pan-Tompkins envelope of `signal` (e.g. `preprocessing.pipeline.
        derive_qrs_envelope` output), required when `config.temporal` is set
        (used for heart-rate features).
    lead_idx : int, optional
        Lead used for heart-rate/wave-delineation features, by default 1 (lead "II").

    Returns
    -------
    pd.DataFrame
        One row per record, `RangeIndex`-indexed (caller reindexes to e.g. `ecg_id`).
    """
    frames: list[pd.DataFrame] = []

    if config.spectral:
        frames.append(_per_lead_frame(dominant_frequency(signal, fs=fs, axis=1), "DF"))
        frames.append(
            _per_lead_frame(spectral_entropy(signal, fs=fs, axis=1), "SpecEn")
        )

        wave_features = wave_spectral_features(signal, fs=fs, lead_idx=lead_idx)
        wave_frames = [
            _per_lead_frame(wave_features[key], prefix)
            for key, prefix in _WAVE_SPECTRAL_PREFIXES.items()
        ]
        wave_frames.append(pd.DataFrame({"n_beats": wave_features["n_beats"]}))
        frames.extend(wave_frames)

    if config.temporal:
        if qrs_envelope is None:
            raise ValueError("config.temporal requires qrs_envelope")

        time_features = time_domain_features(signal, axis=1)
        frames.extend(
            _per_lead_frame(values, _TIME_DOMAIN_PREFIXES[key])
            for key, values in time_features.items()
        )

        hr_features = heart_rate_features(qrs_envelope[:, :, lead_idx], fs=fs)
        frames.append(pd.DataFrame(hr_features))

    if config.wavelet:
        for level_features in wavelet_features(signal, axis=1):
            frames.extend(
                _per_lead_frame(values, key) for key, values in level_features.items()
            )

    return pd.concat(frames, axis=1)
