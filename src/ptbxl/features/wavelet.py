import numpy as np
import pywt
from scipy import stats


def wavelet_features(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    axis: int = 0,
) -> list[dict[str, np.ndarray]]:
    """
    Extracts wavelet features for the given ECG signal using the specified wavelet
    and decomposition level.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    wavelet : str, optional
        Mother wavelet, by default 'db4'
    level : int, optional
        Decomposition level, by default 4
    axis : int, optional
        Axis along whcich the wavelet decompositions is performed, by default 0

    Returns
    -------
    features : list[dict[str, np.ndarray]]
        Extracted wavelet features, one dict per decomposition level:
        energy (nrg), mean, std, skew, kurtosis (kurt), Shannon entropy
        of the coefficients' normalized energy distribution (entropy),
        and the max-magnitude coefficient (dom).
    """

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, axis=axis)

    features = []

    for i, coeff in enumerate(coeffs):
        energy = coeff**2
        total_energy = np.sum(energy, axis=axis, keepdims=True)
        out = np.zeros_like(energy)
        prob = np.divide(energy, total_energy, out=out, where=(total_energy != 0))
        log2p = np.log2(prob, out=np.zeros_like(prob), where=(prob > 0))
        entropy = -np.sum(prob * log2p, axis=axis)

        level_features = {
            f"l{i+1}nrg": np.sum(energy, axis=axis),
            f"l{i+1}mean": np.mean(coeff, axis=axis),
            f"l{i+1}std": np.std(coeff, axis=axis),
            f"l{i+1}skew": stats.skew(coeff, axis=axis),
            f"l{i+1}kurt": stats.kurtosis(coeff, axis=axis),
            f"l{i+1}entropy": entropy,
            f"l{i+1}dom": np.max(np.abs(coeff), axis=axis),
        }
        features.append(level_features)

    return features
