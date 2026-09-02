from typing import Optional

import neurokit2 as nk
import numpy as np
from scipy import signal, stats
from tqdm import tqdm
import pywt


class BuildFeatures:
    """
    Class to build features from a signal
    """
    
    def power_spectral_density(
        self,
        x: np.ndarray,
        fs: float,
        method: str = 'fft',
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

        # Compute power spectral density
        Pxx, freqs = self.power_spectral_density(
            x, fs, method=method, nperseg=nperseg, axis=axis
        )

        # Find index of maximum PSD value
        max_psd_idx = np.argmax(Pxx, axis=axis)
        dominant_freq = freqs[max_psd_idx]

        return np.asarray(dominant_freq)
    
    def spectral_entropy(
        self,
        x: np.ndarray,
        fs: float,
        method: str = 'fft',
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
        
        # Compute power spectral density
        Pxx, _ = self.power_spectral_density(
            x, fs, method, nperseg, axis
        )
        
        # Compute the probability distribution of the PSD
        out = np.ones(Pxx.shape)
        sumpxx = np.sum(Pxx, axis=axis, keepdims=True)
        psd_normalized = np.divide(
            Pxx,
            sumpxx,
            out=out,
            where=(sumpxx != 0),
        )
        
        # Compute the entropy
        out = np.ones(psd_normalized.shape)
        log2pxx = np.log2(psd_normalized, out=out, where=(psd_normalized > 0))
        entropy = -np.sum(psd_normalized * log2pxx, axis=axis)

        return np.asarray(entropy)
    
    def wavelet_features(
        self,
        x: np.ndarray,
        wavelet: str = 'db4',
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

        # Decompose the signal using the wavelet transform
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, axis=axis)

        # Initialize list to store features
        features = []

        # Loop through approximation and detail coefficients
        for i, coeff in enumerate(coeffs):
            energy = coeff ** 2
            total_energy = np.sum(energy, axis=axis, keepdims=True)
            out = np.zeros_like(energy)
            prob = np.divide(energy, total_energy, out=out, where=(total_energy != 0))
            log2p = np.log2(prob, out=np.zeros_like(prob), where=(prob > 0))
            entropy = -np.sum(prob * log2p, axis=axis)

            level_features = {
                f'l{i+1}nrg': np.sum(energy, axis=axis),          # Energy
                f'l{i+1}mean': np.mean(coeff, axis=axis),         # Mean
                f'l{i+1}std': np.std(coeff, axis=axis),           # Std dev
                f'l{i+1}skew': stats.skew(coeff, axis=axis),      # Skewness
                f'l{i+1}kurt': stats.kurtosis(coeff, axis=axis),  # Kurtosis
                f'l{i+1}entropy': entropy,                        # Shannon entropy
                f'l{i+1}dom': np.max(np.abs(coeff), axis=axis),   # Dominant coeff
            }
            features.append(level_features)

        return features

    def time_domain_features(
        self,
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
            'std': np.std(x, axis=axis),
            'rms': np.sqrt(np.mean(x ** 2, axis=axis)),
            'ptp': np.ptp(x, axis=axis),
            'skew': stats.skew(x, axis=axis),
            'kurtosis': stats.kurtosis(x, axis=axis),
            'zcr': np.sum(sign_changes, axis=axis) / n,
        }

    def heart_rate_features(
        self,
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
                rmssd[i] = np.sqrt(np.mean(rr_diff ** 2))
                pnn50[i] = np.mean(np.abs(rr_diff) > 0.05)

        return {
            'heart_rate': heart_rate,
            'rr_mean': rr_mean,
            'rr_std': rr_std,
            'rr_cv': rr_cv,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'n_peaks': n_peaks,
        }

    def wave_spectral_features(
        self,
        x: np.ndarray,
        fs: float,
        lead_idx: int,
        method: str = 'dwt',
    ) -> dict[str, np.ndarray]:
        """
        Extracts per-wave (P, QRS, T) spectral descriptors from a 12-lead
        ECG tensor. Delineation (neurokit2) runs once per record, on a
        single representative lead; the resulting onset/offset sample
        indices are then reused to slice all leads at once (leads are
        time-synchronized), so this stays 12x cheaper than delineating
        every lead. Each wave type's segments from all beats in a record
        are concatenated per lead and reduced with the existing
        `dominant_frequency`/`spectral_entropy` methods in one call, rather
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
        wave_names = ('P', 'QRS', 'T')
        # neurokit2's dwt delineation labels QRS boundaries as
        # "R_Onsets"/"R_Offsets", not "QRS_Onsets"/"QRS_Offsets".
        wave_keys = {
            'P': ('ECG_P_Onsets', 'ECG_P_Offsets'),
            'QRS': ('ECG_R_Onsets', 'ECG_R_Offsets'),
            'T': ('ECG_T_Onsets', 'ECG_T_Offsets'),
        }

        dominant_freq = {w: np.full((m, n_leads), np.nan) for w in wave_names}
        spectral_ent = {w: np.full((m, n_leads), np.nan) for w in wave_names}
        n_beats = np.zeros(m, dtype=int)

        for i in tqdm(range(m), desc='Delineating ECG waves'):
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

            n_beats[i] = len(rpeaks.get('ECG_R_Peaks', []))
            if n_beats[i] == 0:
                continue

            for wave in wave_names:
                onset_key, offset_key = wave_keys[wave]
                onsets = np.asarray(waves.get(onset_key, []), dtype=float)
                offsets = np.asarray(waves.get(offset_key, []), dtype=float)
                valid = ~np.isnan(onsets) & ~np.isnan(offsets)
                onsets = onsets[valid].astype(int)
                offsets = offsets[valid].astype(int)

                segments = [
                    x[i, on:off, :] for on, off in zip(onsets, offsets) if off > on
                ]
                if not segments:
                    continue

                # ponytail: concatenating beats introduces spectral leakage
                # at each segment boundary. Upgrade path if this measurably
                # biases P/T dominant-frequency estimates: taper each
                # segment before concatenation, or average per-beat
                # periodograms instead of concatenating.
                concat = np.concatenate(segments, axis=0)  # (total_samples, n_leads)

                dominant_freq[wave][i, :] = self.dominant_frequency(
                    concat, fs=fs, axis=0
                )
                spectral_ent[wave][i, :] = self.spectral_entropy(concat, fs=fs, axis=0)

        out: dict[str, np.ndarray] = {'n_beats': n_beats}
        for wave in wave_names:
            out[f'{wave}_dominant_freq'] = dominant_freq[wave]
            out[f'{wave}_spectral_entropy'] = spectral_ent[wave]

        return out
