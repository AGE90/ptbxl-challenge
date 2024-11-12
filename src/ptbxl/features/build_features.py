import numpy as np
from scipy import signal


class BuildFeatures:
    """
    Class to build features from a signal
    """
    
    def power_spectral_density(
        self,
        x: np.ndarray,
        fs: float,
        method: str = 'fft',
        nperseg: int = None,
        axis: int = 0,
    ) -> float:
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
        np.ndarray
           The power spectral density of the signal

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

        return Pxx, freqs

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
        Pxx, freqs = self.power_spectral_density(
            x, fs, method=method, nperseg=nperseg, axis=axis
        )

        # Find index of maximum PSD value
        max_psd_idx = np.argmax(Pxx, axis=axis)
        dominant_freq = freqs[max_psd_idx]

        return dominant_freq
    
    def spectral_entropy(
        self,
        x: np.ndarray,
        fs: float,
        method: str = 'fft',
        nperseg: int = None,
        axis: int = 0,
    ) -> float:
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
        float
            Spectral entropy of the signal
        """
        
        # Compute power spectral density
        Pxx, _ = self.power_spectral_density(
            x, fs, method, nperseg, axis
        )
        
        # Compute the probability distribution of the PSD
        out = np.ones(Pxx.shape)
        sumpxx = np.sum(Pxx)
        psd_normalized = np.divide(
            Pxx,
            np.sum(Pxx, axis=axis, keepdims=True),
            out=out,
            where=(sumpxx != 0),
        )
        
        # Compute the entropy
        out = np.ones(psd_normalized.shape)
        log2pxx = np.log2(psd_normalized, out=out, where=(psd_normalized > 0))
        entropy = -np.sum(psd_normalized * log2pxx, axis=axis)
        
        return entropy
