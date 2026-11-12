"""Unit tests for the deterministic DSP preprocessing pipeline (`src/dsp/preprocessor.py`).

Verifies:
1. Phase unwrapping and linear least-squares CFO/SFO slope + offset removal across
   symmetric logical OFDM subcarriers `m_k`.
2. Sliding-window Hampel outlier rejection of impulsive spikes.
3. Zero-phase 3rd-order Butterworth bandpass filtering (`lowcut = 0.1 Hz`,
   `highcut = 15.0 Hz`) preserving `0.3 Hz` chest respiration while attenuating
   high-frequency (`> 20 Hz`) noise and DC offset.
4. Short-Time Fourier Transform (STFT) magnitude spectrogram output shapes.
5. Per-window, per-subcarrier Z-score normalization statistics (zero mean, unit std).
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.dsp.preprocessor import SignalPreprocessor


class TestSignalPreprocessor(unittest.TestCase):
    """Test suite for `SignalPreprocessor` mathematical invariants."""

    def setUp(self) -> None:
        """Set up deterministic RNG and default `SignalPreprocessor` (`K = 52`)."""
        self.rng = np.random.default_rng(42)
        self.num_subcarriers = 52
        self.num_packets = 200
        self.fs = 100.0
        self.preprocessor = SignalPreprocessor(
            num_subcarriers=self.num_subcarriers,
            fs=self.fs,
            hampel_window_size=7,
            hampel_n_sigmas=3.0,
            hampel_mad_scale=1.4826,
            lowcut_hz=0.1,
            highcut_hz=15.0,
            filter_order=3,
            stft_n_fft=64,
            stft_win_length=32,
            stft_hop_length=8,
            norm_eps=1e-8,
        )

    def test_phase_sanitization_removes_linear_cfo_slope_and_offset(self) -> None:
        """Verify linear CFO slope `a_t * m_k` and constant offset `b_t` are removed."""
        m_k = self.preprocessor.subcarrier_indices[:, np.newaxis]  # (52, 1)
        # True non-linear multipath phase profile across subcarriers (quadratic curvature)
        norm_k = np.linspace(-1.0, 1.0, self.num_subcarriers)[:, np.newaxis]
        true_nonlinear_phase = 0.25 * (norm_k**2 - np.mean(norm_k**2))
        true_phase_matrix = np.repeat(true_nonlinear_phase, self.num_packets, axis=1)

        # Inject strong per-packet linear slope a_t and random offset b_t
        slopes = self.rng.uniform(-0.35, 0.35, size=(1, self.num_packets))
        offsets = self.rng.uniform(-np.pi, np.pi, size=(1, self.num_packets))
        corrupted_unwrapped = true_phase_matrix + slopes * m_k + offsets

        # Wrap into [-pi, pi] via complex exponential
        wrapped_phase = np.angle(np.exp(1j * corrupted_unwrapped))
        sanitized_phase = self.preprocessor.sanitize_phase(wrapped_phase)

        # Sanitized phase must have zero linear correlation with m_k and zero mean per packet
        residual_slope = np.sum(m_k * sanitized_phase, axis=0) / np.sum(m_k**2)
        residual_mean = np.mean(sanitized_phase, axis=0)

        np.testing.assert_allclose(residual_slope, 0.0, atol=1e-10)
        np.testing.assert_allclose(residual_mean, 0.0, atol=1e-10)
        np.testing.assert_allclose(sanitized_phase, true_phase_matrix, atol=1e-6)

    def test_hampel_filter_suppresses_impulsive_spikes(self) -> None:
        """Verify isolated high-sigma impulsive spikes are replaced by the local median."""
        t = np.arange(self.num_packets, dtype=np.float64)[np.newaxis, :] / self.fs
        clean_signal = np.sin(2.0 * np.pi * 1.5 * t)
        clean_matrix = np.repeat(clean_signal, self.num_subcarriers, axis=0)

        spiked_matrix = clean_matrix.copy()
        spiked_matrix[0, 50] += 25.0
        spiked_matrix[10, 120] -= 30.0
        spiked_matrix[25, 175] += 18.0

        despiked = self.preprocessor.hampel_filter(spiked_matrix)

        self.assertLess(abs(despiked[0, 50] - clean_matrix[0, 50]), 0.25)
        self.assertLess(abs(despiked[10, 120] - clean_matrix[10, 120]), 0.25)
        self.assertLess(abs(despiked[25, 175] - clean_matrix[25, 175]), 0.25)

    def test_butterworth_bandpass_preserves_respiration_and_attenuates_high_freq(
        self,
    ) -> None:
        """Verify 0.3 Hz respiration passes while 30 Hz noise and DC are attenuated."""
        # Use a 10-second signal (1000 samples @ 100 Hz) for crisp spectral resolution
        num_t = 1000
        t = np.arange(num_t, dtype=np.float64)[np.newaxis, :] / self.fs

        dc_component = 5.0
        resp_03hz = 1.0 * np.sin(2.0 * np.pi * 0.3 * t)
        hf_noise_30hz = 1.0 * np.sin(2.0 * np.pi * 30.0 * t)

        composite = np.repeat(
            dc_component + resp_03hz + hf_noise_30hz,
            self.num_subcarriers,
            axis=0,
        )
        filtered = self.preprocessor.butterworth_bandpass(composite)

        # Evaluate steady-state middle segment over an exact integer number of 0.3 Hz cycles
        # (2 cycles = 2 / 0.3 s = 6.6667 s ≈ 667 samples)
        mid_slice = slice(166, 833)
        resp_ref = resp_03hz[0, mid_slice]
        filt_mid = filtered[0, mid_slice]

        # 1. DC offset (5.0) must be eliminated across the window
        self.assertLess(abs(float(np.mean(filtered[0]))), 0.05)

        # 2. 0.3 Hz respiration amplitude must be preserved (> 85% energy retained)
        corr = float(np.corrcoef(resp_ref, filt_mid)[0, 1])
        self.assertGreater(corr, 0.98)
        rms_ratio = float(np.std(filt_mid) / np.std(resp_ref))
        self.assertGreater(rms_ratio, 0.85)
        self.assertLess(rms_ratio, 1.15)

        # 3. Pure 30 Hz signal (> 20 Hz noise) must be strongly attenuated (< 10% amplitude)
        pure_hf = np.repeat(hf_noise_30hz, self.num_subcarriers, axis=0)
        filt_hf = self.preprocessor.butterworth_bandpass(pure_hf)
        hf_rms_ratio = float(np.std(filt_hf[0, mid_slice]) / np.std(hf_noise_30hz[0, mid_slice]))
        self.assertLess(hf_rms_ratio, 0.10)

    def test_stft_spectrogram_shape(self) -> None:
        """Verify `compute_stft` produces `(K, 33, 26)` magnitude spectrograms."""
        dummy_signal = self.rng.normal(0.0, 1.0, size=(self.num_subcarriers, self.num_packets))
        stft_np = self.preprocessor.compute_stft(dummy_signal)
        self.assertEqual(stft_np.shape, (self.num_subcarriers, 33, 26))
        self.assertTrue(np.all(stft_np >= 0.0))

        # Also test with a PyTorch Tensor input of shape (2, K, T)
        tensor_in = torch.from_numpy(
            self.rng.normal(0.0, 1.0, size=(2, self.num_subcarriers, self.num_packets)).astype(
                np.float32
            )
        )
        stft_from_tensor = self.preprocessor.compute_stft(tensor_in)
        self.assertEqual(stft_from_tensor.shape, (2, self.num_subcarriers, 33, 26))

    def test_end_to_end_process_and_zscore_normalization_stats(self) -> None:
        """Verify `process(H)` returns `(2, K, T)` with zero mean and unit std across `t`."""
        t = np.arange(self.num_packets, dtype=np.float64)[np.newaxis, :] / self.fs
        k_idx = np.arange(self.num_subcarriers, dtype=np.float64)[:, np.newaxis]
        # Include non-linear subcarrier phase variation so sanitized phase has non-zero variance
        cfr_complex = (1.0 + 0.3 * np.sin(2.0 * np.pi * 2.5 * t + 0.1 * k_idx)) * np.exp(
            1j
            * (
                0.8 * np.cos(2.0 * np.pi * 2.5 * t) * np.sin(0.15 * k_idx)
                + 0.05 * (k_idx**2)
            )
        )

        out_tensor = self.preprocessor.process(cfr_complex)
        self.assertIsInstance(out_tensor, torch.Tensor)
        self.assertEqual(out_tensor.dtype, torch.float32)
        self.assertEqual(tuple(out_tensor.shape), (2, self.num_subcarriers, self.num_packets))

        # Check per-channel, per-subcarrier mean ≈ 0 and population std ≈ 1 along time axis=-1
        out_np = out_tensor.numpy()
        means = np.mean(out_np, axis=-1)
        stds = np.std(out_np, axis=-1)

        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(stds, 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
