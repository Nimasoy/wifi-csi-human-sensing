"""Deterministic Digital Signal Processing (DSP) pipeline for Wi-Fi CSI.

This module implements the sequential signal preprocessing stages for complex
Channel Frequency Response (CFR) tensors H in C^{K x T}:
1. Raw amplitude and wrapped phase extraction.
2. Phase sanitization via subcarrier unwrapping and linear least-squares
   removal of Carrier Frequency Offset (CFO) / Sampling Frequency Offset (SFO)
   phase slope and constant phase offset across symmetric OFDM subcarriers.
3. Hampel identifier & outlier replacement along the temporal axis per subcarrier.
4. Zero-phase 3rd-order Butterworth bandpass filtering (0.1 - 15.0 Hz) to retain
   subtle respiration (0.2 - 0.4 Hz) and dynamic human motion Doppler components.
5. Short-Time Fourier Transform (STFT) spectrogram computation utility.
6. Tensor stacking and per-window, per-subcarrier Z-score normalization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
from numpy.lib.stride_tricks import sliding_window_view
import torch


class SignalPreprocessor:
    """Deterministic DSP preprocessor for complex Wi-Fi CSI windows.

    Attributes:
        num_subcarriers: Number of OFDM subcarriers K (must be even, e.g., 52 or 90).
        fs: Packet sampling rate in Hz (default: 100.0 Hz).
        hampel_window_size: Sliding window length for the Hampel filter (default: 7).
        hampel_n_sigmas: Outlier detection threshold in standard deviations (default: 3.0).
        hampel_mad_scale: Normal-consistent Median Absolute Deviation scale factor (1.4826).
        lowcut_hz: Lower cutoff frequency in Hz for Butterworth bandpass filter (0.1 Hz).
        highcut_hz: Upper cutoff frequency in Hz for Butterworth bandpass filter (15.0 Hz).
        filter_order: Butterworth filter order (default: 3).
        stft_n_fft: FFT size for STFT computation (default: 64).
        stft_win_length: Window length in samples for STFT (default: 32).
        stft_hop_length: Hop length in samples for STFT (default: 8).
        norm_eps: Numerical stability constant for Z-score normalization (default: 1e-8).
        subcarrier_indices: Symmetric logical OFDM subcarrier indices m_k excluding 0.
    """

    def __init__(
        self,
        num_subcarriers: int = 52,
        fs: float = 100.0,
        hampel_window_size: int = 7,
        hampel_n_sigmas: float = 3.0,
        hampel_mad_scale: float = 1.4826,
        lowcut_hz: float = 0.1,
        highcut_hz: float = 15.0,
        filter_order: int = 3,
        stft_n_fft: int = 64,
        stft_win_length: int = 32,
        stft_hop_length: int = 8,
        norm_eps: float = 1e-8,
    ) -> None:
        """Initialize the SignalPreprocessor with deterministic DSP parameters.

        Args:
            num_subcarriers: Number of subcarriers K (must be positive and even).
            fs: Sampling rate in Hz.
            hampel_window_size: Window length for temporal Hampel filtering (odd integer).
            hampel_n_sigmas: Threshold multiplier on scaled MAD for outlier replacement.
            hampel_mad_scale: Scale factor converting MAD to Gaussian standard deviation.
            lowcut_hz: High-pass cutoff frequency in Hz (must be > 0 and < highcut_hz).
            highcut_hz: Low-pass cutoff frequency in Hz (must be < fs / 2).
            filter_order: Order of the Butterworth bandpass filter.
            stft_n_fft: Number of FFT points for STFT spectrogram utility.
            stft_win_length: Hann window length for STFT.
            stft_hop_length: Frame hop size for STFT.
            norm_eps: Small epsilon to prevent division by zero during Z-score normalization.

        Raises:
            ValueError: If `num_subcarriers` is not a positive even integer, or if filter
                cutoff frequencies violate the Nyquist criterion.
        """
        if num_subcarriers <= 0 or num_subcarriers % 2 != 0:
            raise ValueError(
                f"num_subcarriers must be a positive even integer, got {num_subcarriers}."
            )
        nyquist = 0.5 * fs
        if not (0.0 < lowcut_hz < highcut_hz < nyquist):
            raise ValueError(
                f"Invalid bandpass cutoffs: require 0 < lowcut ({lowcut_hz}) < "
                f"highcut ({highcut_hz}) < Nyquist ({nyquist})."
            )
        if hampel_window_size < 3 or hampel_window_size % 2 == 0:
            raise ValueError(
                f"hampel_window_size must be an odd integer >= 3, got {hampel_window_size}."
            )

        self.num_subcarriers: int = num_subcarriers
        self.fs: float = float(fs)
        self.hampel_window_size: int = hampel_window_size
        self.hampel_n_sigmas: float = float(hampel_n_sigmas)
        self.hampel_mad_scale: float = float(hampel_mad_scale)
        self.lowcut_hz: float = float(lowcut_hz)
        self.highcut_hz: float = float(highcut_hz)
        self.filter_order: int = filter_order
        self.stft_n_fft: int = stft_n_fft
        self.stft_win_length: int = stft_win_length
        self.stft_hop_length: int = stft_hop_length
        self.norm_eps: float = float(norm_eps)

        self.subcarrier_indices: NDArray[np.float64] = (
            self._build_symmetric_subcarrier_indices(self.num_subcarriers)
        )

        # Precompute 3rd-order Butterworth bandpass filter coefficients (b, a)
        self._b_butter: NDArray[np.float64]
        self._a_butter: NDArray[np.float64]
        self._b_butter, self._a_butter = sp_signal.butter(
            N=self.filter_order,
            Wn=[self.lowcut_hz, self.highcut_hz],
            btype="bandpass",
            fs=self.fs,
        )

    @staticmethod
    def _build_symmetric_subcarrier_indices(num_subcarriers: int) -> NDArray[np.float64]:
        """Generate symmetric logical OFDM subcarrier indices m_k excluding DC (0).

        For any even K (e.g., K=52 for IEEE 802.11n 20 MHz or K=90 for Widar3.0),
        generates [-K/2, ..., -1, 1, ..., K/2].

        Args:
            num_subcarriers: Positive even integer K.

        Returns:
            1D float64 NumPy array of shape (K,) containing symmetric indices.
        """
        half_k = num_subcarriers // 2
        neg_indices = np.arange(-half_k, 0, dtype=np.float64)
        pos_indices = np.arange(1, half_k + 1, dtype=np.float64)
        return np.concatenate([neg_indices, pos_indices], axis=0)

    @staticmethod
    def extract_amplitude_and_phase(
        csi_complex: NDArray[np.complexfloating],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute raw amplitude A = |H| and wrapped phase phi_raw = angle(H).

        Args:
            csi_complex: Complex CFR matrix H of shape (K, T).

        Returns:
            Tuple of (amplitude, wrapped_phase), each a float64 array of shape (K, T)
            with wrapped_phase in [-pi, pi].
        """
        amplitude = np.abs(csi_complex).astype(np.float64)
        wrapped_phase = np.angle(csi_complex).astype(np.float64)
        return amplitude, wrapped_phase

    def sanitize_phase(
        self,
        wrapped_phase: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Unwrap phase along subcarriers and remove linear timing/CFO phase errors.

        For each time packet t, unwraps phase along the subcarrier axis k (axis=0)
        and fits a linear model across symmetric logical subcarrier indices m_k:
            phi_unwrap(k, t) ≈ a_t * m_k + b_t
        Because m_k is strictly symmetric around 0:
            a_t = (sum_k m_k * phi_unwrap(k, t)) / (sum_k m_k^2)
            b_t = (1 / K) * sum_k phi_unwrap(k, t)
        The sanitized phase is phi_hat(k, t) = phi_unwrap(k, t) - (a_t * m_k + b_t).

        Args:
            wrapped_phase: Wrapped phase array of shape (K, T) in radians.

        Returns:
            Sanitized phase array phi_hat of shape (K, T) in float64.

        Raises:
            ValueError: If the first dimension of `wrapped_phase` does not match K.
        """
        num_k = wrapped_phase.shape[0]
        if num_k != self.num_subcarriers:
            m_k = self._build_symmetric_subcarrier_indices(num_k)
        else:
            m_k = self.subcarrier_indices

        unwrapped = np.unwrap(wrapped_phase, axis=0)

        # m_k has shape (K, 1) for vectorized projection across all T packets
        m_col = m_k[:, np.newaxis]
        m_sum_sq = float(np.sum(m_k**2))

        # Linear slope a_t of shape (1, T) and constant offset b_t of shape (1, T)
        slope_a = np.sum(m_col * unwrapped, axis=0, keepdims=True) / m_sum_sq
        offset_b = np.mean(unwrapped, axis=0, keepdims=True)

        sanitized = unwrapped - (slope_a * m_col + offset_b)
        return sanitized.astype(np.float64)

    def hampel_filter(
        self,
        data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply sliding-window Hampel outlier rejection along the time axis (axis=1).

        Operates independently per subcarrier k across time t using window size W,
        replacing any sample x(k, t) where |x(k, t) - median| > n_sigmas * 1.4826 * MAD
        with the local window median.

        Args:
            data: 2D float64 array of shape (K, T).

        Returns:
            Despiked 2D float64 array of shape (K, T).
        """
        half_win = self.hampel_window_size // 2
        # Edge-pad along time axis (axis=1) to preserve boundary length T
        padded = np.pad(data, pad_width=((0, 0), (half_win, half_win)), mode="edge")
        # windows shape: (K, T, window_size)
        windows = sliding_window_view(
            padded, window_shape=self.hampel_window_size, axis=1
        )

        medians = np.median(windows, axis=-1)
        abs_dev = np.abs(windows - medians[..., np.newaxis])
        mad = np.median(abs_dev, axis=-1)
        sigma_est = self.hampel_mad_scale * mad

        diff = np.abs(data - medians)
        outlier_mask = diff > (self.hampel_n_sigmas * sigma_est)

        filtered = np.where(outlier_mask, medians, data)
        return filtered.astype(np.float64)

    def butterworth_bandpass(
        self,
        data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply zero-phase 3rd-order Butterworth bandpass filtering along time axis.

        Centers each subcarrier across the window (`data - mean(data, axis=1)`) to
        prevent large DC step transients over short windows, then applies
        `scipy.signal.filtfilt` with `lowcut = 0.1 Hz` and `highcut = 15.0 Hz`
        at `fs = 100.0 Hz` along axis=1 so that both chest respiration (0.2-0.4 Hz)
        and torso/limb motion frequencies are preserved with zero phase distortion.

        Args:
            data: 2D float64 array of shape (K, T).

        Returns:
            Bandpass-filtered 2D float64 array of shape (K, T).
        """
        centered = data - np.mean(data, axis=1, keepdims=True)
        filtered = sp_signal.filtfilt(
            self._b_butter,
            self._a_butter,
            centered,
            axis=1,
            padtype="even",
        )
        return np.asarray(filtered, dtype=np.float64)

    def compute_stft(
        self,
        signal_input: NDArray[np.float64] | torch.Tensor,
    ) -> NDArray[np.float64]:
        """Compute Short-Time Fourier Transform (STFT) magnitude spectrogram.

        Uses `scipy.signal.stft` with Hann window, `n_fft = 64`, `win_length = 32`,
        and `hop_length = 8` (`noverlap = 24`) along the last axis (time axis).

        Args:
            signal_input: 1D array `(T,)`, 2D array `(K, T)`, or 3D array `(C, K, T)`
                (accepts either NumPy ndarray or PyTorch Tensor).

        Returns:
            STFT magnitude spectrogram `|Zxx|` of shape `(..., F, N_frames)` as a
            float64 NumPy array, where `F = n_fft // 2 + 1 = 33`.
        """
        if isinstance(signal_input, torch.Tensor):
            sig_np = signal_input.detach().cpu().numpy().astype(np.float64)
        else:
            sig_np = np.asarray(signal_input, dtype=np.float64)

        noverlap = self.stft_win_length - self.stft_hop_length
        _, _, zxx = sp_signal.stft(
            sig_np,
            fs=self.fs,
            window="hann",
            nperseg=self.stft_win_length,
            noverlap=noverlap,
            nfft=self.stft_n_fft,
            axis=-1,
        )
        return np.abs(zxx).astype(np.float64)

    def normalize_window(
        self,
        stacked_tensor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply per-window, per-channel, per-subcarrier Z-score normalization across time.

        Args:
            stacked_tensor: Array of shape `(2, K, T)` containing `[A_filt, phi_filt]`.

        Returns:
            Normalized float64 array of shape `(2, K, T)` with zero mean and unit
            variance along the time axis `t` (axis=-1).
        """
        mean = np.mean(stacked_tensor, axis=-1, keepdims=True)
        std = np.std(stacked_tensor, axis=-1, keepdims=True)
        return (stacked_tensor - mean) / (std + self.norm_eps)

    def process(
        self,
        csi_complex: NDArray[np.complexfloating],
    ) -> torch.Tensor:
        """Execute the full deterministic DSP preprocessing cascade on a CFR window.

        Pipeline steps:
            1. Compute raw amplitude `A = |H|` and wrapped phase `phi_raw = angle(H)`.
            2. Phase sanitization (`unwrap` + linear CFO/SFO removal across `m_k`).
            3. Temporal Hampel outlier filtering on both `A` and `phi_hat`.
            4. Zero-phase 3rd-order Butterworth bandpass filtering (0.1 - 15.0 Hz).
            5. Stack `[A_filt, phi_filt]` into `(2, K, T)` and Z-score normalize.

        Args:
            csi_complex: Complex Channel Frequency Response matrix `H` of shape `(K, T)`.

        Returns:
            Preprocessed PyTorch float32 tensor of shape `(2, K, T)`.

        Raises:
            ValueError: If `csi_complex` is not a 2D array with `K == num_subcarriers`.
        """
        if csi_complex.ndim != 2 or csi_complex.shape[0] != self.num_subcarriers:
            raise ValueError(
                f"Expected complex CSI matrix of shape ({self.num_subcarriers}, T), "
                f"got {csi_complex.shape}."
            )

        # 1. Raw amplitude and wrapped phase
        amp_raw, phase_raw = self.extract_amplitude_and_phase(csi_complex)

        # 2. Phase sanitization
        phase_sanitized = self.sanitize_phase(phase_raw)

        # 3. Hampel outlier filtering along time axis t
        amp_despiked = self.hampel_filter(amp_raw)
        phase_despiked = self.hampel_filter(phase_sanitized)

        # 4. 3rd-order zero-phase Butterworth bandpass filtering (0.1 - 15.0 Hz)
        amp_filt = self.butterworth_bandpass(amp_despiked)
        phase_filt = self.butterworth_bandpass(phase_despiked)

        # 5. Stack into (2, K, T) and apply per-subcarrier Z-score normalization
        stacked = np.stack([amp_filt, phase_filt], axis=0)
        normalized = self.normalize_window(stacked)

        return torch.from_numpy(normalized.astype(np.float32))

    def __call__(
        self,
        csi_complex: NDArray[np.complexfloating],
    ) -> torch.Tensor:
        """Callable alias for `process(csi_complex)`."""
        return self.process(csi_complex)
