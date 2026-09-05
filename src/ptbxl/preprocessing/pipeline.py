"""Composed preprocessing stages, driven by `ptbxl.config.PreprocessingConfig`."""

import numpy as np

from ptbxl.config import PreprocessingConfig
from ptbxl.preprocessing.filtering import mean_removal, pan_tompkins, wander_removal


def preprocess_ecg(
    signal: np.ndarray, fs: float, config: PreprocessingConfig
) -> np.ndarray:
    """Wander removal (fixed fc=0.5Hz) followed by mean removal.

    Matches `ecg_signals_m0` in notebooks 02/03: the real-morphology signal used
    by feature extraction (as opposed to the QRS envelope from `derive_qrs_envelope`).
    """
    signal = wander_removal(signal, fs, axis=1)
    signal = mean_removal(signal, axis=1)
    return signal


def derive_qrs_envelope(
    signal: np.ndarray, fs: float, config: PreprocessingConfig
) -> np.ndarray:
    """Pan-Tompkins QRS envelope, driven by `config.{lowcut,highcut,filter_order}`.

    Matches `ecg_signals_pt` in notebooks 02/03. Only used for heart-rate feature
    extraction — real ECG morphology (from `preprocess_ecg`) is needed everywhere else.
    """
    w = int((config.window_ms / 1000) * fs)
    return pan_tompkins(
        signal,
        fs,
        w,
        fc_low=config.lowcut,
        fc_high=config.highcut,
        order=config.filter_order,
        axis=1,
    )
