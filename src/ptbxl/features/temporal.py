import numpy as np
from scipy import signal, stats


def time_domain_features(
    x: np.ndarray,
    axis: int = 0,
) -> dict[str, np.ndarray]:
    """
    Extracts time-domain morphology features from an ECG signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    axis : int, optional
        Axis along which the signal is reduced, by default 0

    Returns
    -------
    features : dict
        std, rms, ptp (peak-to-peak), skew, kurtosis and zcr
        (zero-crossing rate), each reduced along `axis`. Excludes mean:
        for mean-removed input (e.g. `ecg_signals_m0`) it is ~0 by
        construction along the same axis, carrying no information.
    """

    n = x.shape[axis]
    sign_changes = np.diff(np.sign(x), axis=axis) != 0

    return {
        "std": np.std(x, axis=axis),
        "rms": np.sqrt(np.mean(x**2, axis=axis)),
        "ptp": np.ptp(x, axis=axis),
        "skew": stats.skew(x, axis=axis),
        "kurtosis": stats.kurtosis(x, axis=axis),
        "zcr": np.sum(sign_changes, axis=axis) / n,
    }


def heart_rate_features(
    x: np.ndarray,
    fs: float,
    height_frac: float = 0.3,
    min_rr_sec: float = 0.3,
) -> dict[str, np.ndarray]:
    """
    Extracts heart-rate / RR-interval features from a QRS-envelope
    signal (e.g. the Pan-Tompkins moving-window-integrated output of a
    single lead) via R-peak detection.

    Parameters
    ----------
    x : np.ndarray
        QRS-envelope signal, shape (records, samples).
    fs : float
        Sampling frequency of the signal in Hz.
    height_frac : float, optional
        Peak detection height threshold, as a fraction of each record's
        max value, by default 0.3
    min_rr_sec : float, optional
        Minimum spacing between accepted peaks, in seconds
        (caps the detectable heart rate), by default 0.3

    Returns
    -------
    features : dict
        heart_rate, rr_mean, rr_std, rr_cv, rmssd, pnn50, n_peaks, each
        an array of length `records`. NaN where too few R-peaks are
        detected (rmssd/pnn50 need >=3 peaks, the rest need >=2).
    """

    m = x.shape[0]
    min_rr_samples = int(min_rr_sec * fs)

    heart_rate = np.full(m, np.nan)
    rr_mean = np.full(m, np.nan)
    rr_std = np.full(m, np.nan)
    rr_cv = np.full(m, np.nan)
    rmssd = np.full(m, np.nan)
    pnn50 = np.full(m, np.nan)
    n_peaks = np.zeros(m, dtype=int)

    for i in range(m):
        sig = x[i, :]
        peaks, _ = signal.find_peaks(
            sig, height=height_frac * sig.max(), distance=min_rr_samples
        )
        n_peaks[i] = len(peaks)
        if len(peaks) < 2:
            continue

        rr = np.diff(peaks) / fs
        heart_rate[i] = 60 / rr.mean()
        rr_mean[i] = rr.mean()
        rr_std[i] = rr.std()
        rr_cv[i] = rr.std() / rr.mean()

        if len(rr) >= 2:
            rr_diff = np.diff(rr)
            rmssd[i] = np.sqrt(np.mean(rr_diff**2))
            pnn50[i] = np.mean(np.abs(rr_diff) > 0.05)

    return {
        "heart_rate": heart_rate,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_cv": rr_cv,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "n_peaks": n_peaks,
    }
