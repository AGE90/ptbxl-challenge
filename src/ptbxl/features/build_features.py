import numpy as np
from scipy import signal


class BuildFeatures:
    """
    Class to build features from a signal
    """

    def dominant_frequency(
        self,
        x: np.ndarray,
        fs: float,
        method: str = 'fft',
        nperseg: int = None,
        axis: int = 0,
    ) -> float:
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
        float
           Dominant frequency of the signal

        Raises
        ------
        ValueError
            If a proper method is not selected
        """

        # Compute power spectral density
        if method == 'fft':
            freqs, Pxx = signal.periodogram(x, fs, axis=axis)
        elif method == 'welch':
            freqs, Pxx = signal.welch(x, fs, nperseg=nperseg, axis=axis)
        else:
            raise ValueError(f'Invalid method: {method}')

        # print(f'freqs {freqs.shape}')
        # print(f'Pxx {Pxx.shape}')

        # Find index of maximum PSD value
        max_psd_idx = np.argmax(Pxx, axis=axis)
        dominant_freq = freqs[max_psd_idx]

        return dominant_freq
