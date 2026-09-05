from typing import Optional

import neurokit2 as nk
import numpy as np
from scipy import signal
from tqdm import tqdm


def power_spectral_density(
    x: np.ndarray,
    fs: float,
    method: str = "fft",
    nperseg: Optional[int] = None,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the power spectral density of a signal

    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : float
        Sampling frequency of the signal in Hz
    method : str, optional
        'fft' or 'welch', by default 'fft'
    nperseg : int, optional
        Length of each segment for Welch's method, by default None
    axis : int
        Axis along whcich the power spectral density is computed, by default 0

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
       The power spectral density of the signal and its frequency bins
       (Pxx, freqs)

    Raises
    ------
    ValueError
        If a proper method is not selected
    """

    if method == "fft":
        freqs, Pxx = signal.periodogram(x, fs, axis=axis)
    elif method == "welch":
        freqs, Pxx = signal.welch(x, fs, nperseg=nperseg, axis=axis)
    else:
        raise ValueError(f"Invalid method: {method}")

    return Pxx, freqs


def dominant_frequency(
    x: np.ndarray,
    fs: float,
    method: str = "fft",
    nperseg: Optional[int] = None,
    axis: int = 0,
) -> np.ndarray:
    """Computes the dominant frequency of a signal

    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : float
        Sampling frequency of the signal in Hz
    method : str, optional
        'fft' or 'welch', by default 'fft'
    nperseg : int, optional
        Length of each segment for Welch's method, by default None
    axis : int
        Axis along whcich the power spectral density is computed, by default 0

    Returns
    -------
    np.ndarray
       Dominant frequency of the signal

    Raises
    ------
    ValueError
        If a proper method is not selected
    """

    Pxx, freqs = power_spectral_density(
        x, fs, method=method, nperseg=nperseg, axis=axis
    )

    max_psd_idx = np.argmax(Pxx, axis=axis)
    dominant_freq = freqs[max_psd_idx]

    return np.asarray(dominant_freq)


def spectral_entropy(
    x: np.ndarray,
    fs: float,
    method: str = "fft",
    nperseg: Optional[int] = None,
    axis: int = 0,
) -> np.ndarray:
    """Computes the spectral entropy of a signal

    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : float
        Sampling frequency of the signal in Hz
    method : str, optional
        'fft' or 'welch', by default 'fft'
    nperseg : int, optional
        Length of each segment for Welch's method, by default None
    axis : int
        Axis along whcich the power spectral density is computed, by default 0

    Returns
    -------
    np.ndarray
        Spectral entropy of the signal
    """

    Pxx, _ = power_spectral_density(x, fs, method, nperseg, axis)

    out = np.ones(Pxx.shape)
    sumpxx = np.sum(Pxx, axis=axis, keepdims=True)
    psd_normalized = np.divide(Pxx, sumpxx, out=out, where=(sumpxx != 0))

    out = np.ones(psd_normalized.shape)
    log2pxx = np.log2(psd_normalized, out=out, where=(psd_normalized > 0))
    entropy = -np.sum(psd_normalized * log2pxx, axis=axis)

    return np.asarray(entropy)


def wave_spectral_features(
    x: np.ndarray,
    fs: float,
    lead_idx: int,
    method: str = "dwt",
) -> dict[str, np.ndarray]:
    """
    Extracts per-wave (P, QRS, T) spectral descriptors from a 12-lead
    ECG tensor. Delineation (neurokit2) runs once per record, on a
    single representative lead; the resulting onset/offset sample
    indices are then reused to slice all leads at once (leads are
    time-synchronized), so this stays 12x cheaper than delineating
    every lead. Each wave type's segments from all beats in a record
    are concatenated per lead and reduced with the existing
    `dominant_frequency`/`spectral_entropy` functions in one call, rather
    than re-implementing PSD/entropy math per beat.

    Parameters
    ----------
    x : np.ndarray
        ECG signal, shape (records, samples, leads). Pass the wander-
        and mean-removed signal (e.g. `ecg_signals_m0`), NOT the
        Pan-Tompkins envelope (`ecg_signals_pt`) -- delineation needs
        real ECG morphology, which the envelope destroys.
    fs : float
        Sampling frequency of the signal in Hz.
    lead_idx : int
        Index of the lead to delineate (e.g. `leads.index("II")`,
        matching the convention already used for `heart_rate_features`).
    method : str, optional
        neurokit2 `ecg_delineate` method, by default 'dwt' (needed for
        onset/offset boundaries).

    Returns
    -------
    features : dict
        'n_beats' (records,) plus, for each wave in {P, QRS, T}:
        '{wave}_dominant_freq' and '{wave}_spectral_entropy', each
        (records, leads). NaN-filled per-record where delineation finds
        no beats of that wave type, or where neurokit2 fails on a
        pathological record.
    """

    m, _, n_leads = x.shape
    wave_names = ("P", "QRS", "T")
    # neurokit2's dwt delineation labels QRS boundaries as
    # "R_Onsets"/"R_Offsets", not "QRS_Onsets"/"QRS_Offsets".
    wave_keys = {
        "P": ("ECG_P_Onsets", "ECG_P_Offsets"),
        "QRS": ("ECG_R_Onsets", "ECG_R_Offsets"),
        "T": ("ECG_T_Onsets", "ECG_T_Offsets"),
    }

    dominant_freq = {w: np.full((m, n_leads), np.nan) for w in wave_names}
    spectral_ent = {w: np.full((m, n_leads), np.nan) for w in wave_names}
    n_beats = np.zeros(m, dtype=int)

    for i in tqdm(range(m), desc="Delineating ECG waves"):
        lead_sig = x[i, :, lead_idx]

        try:
            # neurokit2 ships no py.typed marker, and its ecg_peaks /
            # ecg_delineate submodules re-export a function with the
            # same name as their own module (ecg/ecg_peaks.py defines
            # ecg_peaks()) -- static analyzers without real type info
            # can misresolve the attribute as the submodule itself,
            # reporting "not callable" even though this runs fine.
            _, rpeaks = nk.ecg_peaks(lead_sig, sampling_rate=fs)  # type: ignore
            _, waves = nk.ecg_delineate(  # type: ignore
                lead_sig, rpeaks, sampling_rate=fs, method=method
            )
        except Exception:
            # dwt delineation can raise on pathological signals (too
            # few beats, flat lines, heavy noise), unlike
            # scipy.signal.find_peaks in heart_rate_features, which
            # never raises. Skip this record, leave it NaN.
            continue

        n_beats[i] = len(rpeaks.get("ECG_R_Peaks", []))
        if n_beats[i] == 0:
            continue

        for wave in wave_names:
            onset_key, offset_key = wave_keys[wave]
            onsets = np.asarray(waves.get(onset_key, []), dtype=float)
            offsets = np.asarray(waves.get(offset_key, []), dtype=float)
            valid = ~np.isnan(onsets) & ~np.isnan(offsets)
            onsets = onsets[valid].astype(int)
            offsets = offsets[valid].astype(int)

            segments = [x[i, on:off, :] for on, off in zip(onsets, offsets) if off > on]
            if not segments:
                continue

            # ponytail: concatenating beats introduces spectral leakage
            # at each segment boundary. Upgrade path if this measurably
            # biases P/T dominant-frequency estimates: taper each
            # segment before concatenation, or average per-beat
            # periodograms instead of concatenating.
            concat = np.concatenate(segments, axis=0)  # (total_samples, n_leads)

            dominant_freq[wave][i, :] = dominant_frequency(concat, fs=fs, axis=0)
            spectral_ent[wave][i, :] = spectral_entropy(concat, fs=fs, axis=0)

    out: dict[str, np.ndarray] = {"n_beats": n_beats}
    for wave in wave_names:
        out[f"{wave}_dominant_freq"] = dominant_freq[wave]
        out[f"{wave}_spectral_entropy"] = spectral_ent[wave]

    return out
