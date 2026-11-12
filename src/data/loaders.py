"""Hardware CSI dataset loaders for Widar3.0 (.dat) and ESP32-CSI-Tool (.csv).

Provides PyTorch `Dataset` implementations that parse raw hardware CSI recordings,
slice them into sliding windows of length `T = 200` with `50%` stride (`100` packets),
and execute `SignalPreprocessor` to return `(preprocessed_tensor, label)` pairs:
1. `Widar3Dataset`: Parses Intel 5300 Linux 802.11n CSI Tool binary `.dat` files
   (`code = 187 / 0xBB` beamforming records, 30 subcarriers across up to 3 links
   yielding `K = 90` subcarriers by default).
2. `ESP32CSIDataset`: Parses Espressif ESP32-CSI-Tool serial `.csv` logs containing
   interleaved `[imag, real, ...]` or `[real, imag, ...]` int8/float subcarrier arrays
   (`K = 52` active IEEE 802.11n 20 MHz subcarriers after removing DC and guard bands).
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import struct

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from src.data.synthetic_csi import CLASS_LABELS
from src.dsp.preprocessor import SignalPreprocessor

_CLASS_NAME_TO_ID: dict[str, int] = {
    name.lower(): idx for idx, name in CLASS_LABELS.items()
}


def _infer_label_from_path(file_path: Path) -> int:
    """Infer integer class label in `{0, 1, 2, 3, 4}` from file or parent directory name.

    Checks both the parent directory name and the filename stem for class names
    (`Empty`, `Stationary_Presence`, `Walking`, `Sitting_Standing`, `Falling`) or
    explicit numeric prefixes (`class_0`, `label_2`, `0_`, etc.). Defaults to `0`
    if no explicit pattern is matched.

    Args:
        file_path: Path to the `.dat` or `.csv` CSI log file.

    Returns:
        Integer class label in `{0, 1, 2, 3, 4}`.
    """
    candidates = [file_path.parent.name.lower(), file_path.stem.lower()]
    for text in candidates:
        for class_name, class_id in _CLASS_NAME_TO_ID.items():
            if class_name in text:
                return class_id
        if text in {"0", "1", "2", "3", "4"}:
            return int(text)
        match = re.search(r"(?:class|label|act|id)[_-]?([0-4])\b", text)
        if match is not None:
            return int(match.group(1))
        prefix_match = re.match(r"^([0-4])[_-]", text)
        if prefix_match is not None:
            return int(prefix_match.group(1))
    return 0


def _slice_or_pad_subcarriers(
    csi_matrix: NDArray[np.complex128],
    target_k: int,
) -> NDArray[np.complex128]:
    """Ensure a complex CSI matrix `(K_raw, T)` matches `(target_k, T)`.

    If `K_raw == 64` and `target_k == 52` (standard IEEE 802.11n HT20 64-FFT with
    DC and guard bands present), removes the 11 guard subcarriers `[0..5, 59..63]`
    and the DC subcarrier `[32]`, retaining the 52 active data/pilot subcarriers.
    Otherwise, center-crops or tiles/pads along the subcarrier axis to `target_k`.

    Args:
        csi_matrix: Complex128 array of shape `(K_raw, T)`.
        target_k: Desired even subcarrier count `K` (e.g., 52 or 90).

    Returns:
        Complex128 array of shape `(target_k, T)`.
    """
    k_raw, num_packets = csi_matrix.shape
    if k_raw == target_k:
        return csi_matrix

    if k_raw == 64 and target_k == 52:
        neg_active = csi_matrix[6:32, :]
        pos_active = csi_matrix[33:59, :]
        return np.concatenate([neg_active, pos_active], axis=0)

    if k_raw > target_k:
        start = (k_raw - target_k) // 2
        return csi_matrix[start : start + target_k, :]

    reps = (target_k + k_raw - 1) // k_raw
    tiled = np.tile(csi_matrix, (reps, 1))
    return tiled[:target_k, :num_packets]


def _extract_sliding_windows(
    csi_stream: NDArray[np.complex128],
    window_size: int = 200,
    stride: int = 100,
) -> list[NDArray[np.complex128]]:
    """Slice a continuous CFR stream `(K, T_total)` into `(K, window_size)` windows.

    If `T_total < window_size`, edge-pads along the time axis so at least one
    valid window of length `window_size` is returned.

    Args:
        csi_stream: Complex128 CSI stream of shape `(K, T_total)`.
        window_size: Window length `T` in packets (default: 200).
        stride: Step size in packets between consecutive windows (default: 100).

    Returns:
        List of complex128 windows, each of shape `(K, window_size)`.
    """
    total_t = csi_stream.shape[1]
    if total_t < window_size:
        pad_len = window_size - total_t
        padded = np.pad(csi_stream, ((0, 0), (0, pad_len)), mode="edge")
        return [padded]

    windows: list[NDArray[np.complex128]] = []
    for start_t in range(0, total_t - window_size + 1, stride):
        windows.append(csi_stream[:, start_t : start_t + window_size].copy())
    return windows


class Widar3Dataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset for Widar3.0 / Intel 5300 binary `.dat` CSI files.

    Supports both:
        1. Standard Linux 802.11n CSI Tool binary `.dat` records (`code = 0xBB / 187`,
           20-byte header + bit-packed 30-subcarrier signed 8-bit I/Q payload across
           $N_{rx} \times N_{tx}$ antenna pairs).
        2. Compact binary `.dat` matrices prefixed with a `WIDAR3_RAW` magic header
           or raw float32/complex64 streams for fast local testing.

    Attributes:
        root_dir: Path to the directory containing `.dat` files.
        num_subcarriers: Number of subcarriers `K` (default: 90 for $3 \times 30$ links).
        window_size: Number of time packets `T` per window (default: 200).
        stride: Sliding window stride in packets (default: 100).
        fs: Sampling frequency in Hz (default: 100.0).
        preprocessor: `SignalPreprocessor` instance applied to every window.
        windows: List of extracted raw complex CFR windows `H in C^{K x T}`.
        labels: List of integer labels in `{0, 1, 2, 3, 4}` corresponding to `windows`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_subcarriers: int = 90,
        window_size: int = 200,
        stride: int = 100,
        fs: float = 100.0,
        preprocessor: SignalPreprocessor | None = None,
    ) -> None:
        """Initialize the `Widar3Dataset` from a directory of `.dat` CSI logs.

        Args:
            root_dir: Local directory containing Widar3.0 `.dat` files (either directly
                or organized in class subfolders `0_Empty`, `1_Stationary_Presence`, etc.).
            num_subcarriers: Target subcarrier count `K` (default: 90).
            window_size: Packets per window `T` (default: 200).
            stride: Packet stride between windows (default: 100).
            fs: Sampling rate in Hz (default: 100.0).
            preprocessor: Optional custom `SignalPreprocessor`.

        Raises:
            FileNotFoundError: If `root_dir` does not exist or contains no `.dat` files,
                with detailed setup instructions for Widar3.0.
        """
        super().__init__()
        self.root_dir: Path = Path(root_dir)
        self.num_subcarriers: int = num_subcarriers
        self.window_size: int = window_size
        self.stride: int = stride
        self.fs: float = float(fs)
        self.preprocessor: SignalPreprocessor = (
            preprocessor
            if preprocessor is not None
            else SignalPreprocessor(num_subcarriers=num_subcarriers, fs=fs)
        )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Widar3.0 dataset directory not found at '{self.root_dir.resolve()}'.\n"
                "Setup Instructions for Widar3.0 (.dat):\n"
                "  1. Create the directory: mkdir -p data/raw/widar3\n"
                "  2. Download Widar3.0 Intel 5300 CSI (.dat) logs from the official IEEE DataPort / "
                "Tsinghua Widar3.0 repository.\n"
                "  3. Organize `.dat` files under class subdirectories "
                "(e.g., `data/raw/widar3/0_Empty/*.dat`, `data/raw/widar3/2_Walking/*.dat`) "
                "or prefix filenames with the class ID (`0_sample.dat` .. `4_sample.dat`)."
            )

        dat_files = sorted(self.root_dir.rglob("*.dat"))
        if not dat_files:
            raise FileNotFoundError(
                f"No Widar3.0 '.dat' files found inside '{self.root_dir.resolve()}'.\n"
                "Setup Instructions: Place Intel 5300 CSI '.dat' files into "
                f"'{self.root_dir}' (or class subfolders '0_Empty'..'4_Falling') and retry."
            )

        self.windows: list[NDArray[np.complex128]] = []
        self.labels: list[int] = []

        for file_path in dat_files:
            label = _infer_label_from_path(file_path)
            csi_stream = self.parse_dat_file(file_path, target_k=self.num_subcarriers)
            file_windows = _extract_sliding_windows(
                csi_stream,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.windows.extend(file_windows)
            self.labels.extend([label] * len(file_windows))

    @staticmethod
    def parse_dat_file(
        file_path: str | Path,
        target_k: int = 90,
    ) -> NDArray[np.complex128]:
        """Parse an Intel 5300 / Widar3.0 binary `.dat` file into `H in C^{K x T}`.

        Supports standard Linux 802.11n CSI Tool binary frames (`code = 0xBB` / `187`
        with 16-bit big-endian frame length prefix and bit-packed 30-subcarrier signed
        I/Q values) as well as direct `complex64`/`complex128` binary dumps.

        Args:
            file_path: Path to the `.dat` file.
            target_k: Number of subcarriers `K` to output (default: 90).

        Returns:
            Complex128 NumPy array of shape `(target_k, T_packets)`.

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            ValueError: If the binary file is empty or cannot be decoded into CSI packets.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Widar3.0 .dat file not found: '{path}'.")

        raw_bytes = path.read_bytes()
        if len(raw_bytes) < 24:
            raise ValueError(f"Widar3.0 .dat file '{path}' is too small ({len(raw_bytes)} bytes).")

        packets: list[NDArray[np.complex128]] = []
        offset = 0
        total_len = len(raw_bytes)

        # Attempt standard Intel 5300 Linux 802.11n CSI Tool frame parsing (code == 0xBB)
        while offset + 3 <= total_len:
            field_len = struct.unpack(">H", raw_bytes[offset : offset + 2])[0]
            code = raw_bytes[offset + 2]
            if field_len == 0 or offset + 2 + field_len > total_len:
                break

            if code == 187 and field_len >= 21:
                # 19 bytes header after code: timestamp_low(4), bfee_count(2), reserved(2),
                # Nrx(1), Ntx(1), rssi_a(1), rssi_b(1), rssi_c(1), noise(1), agc(1),
                # perm(1), rate(2), csi_len(2) -> payload starts at offset + 22
                hdr_start = offset + 3
                nrx = max(1, raw_bytes[hdr_start + 8])
                ntx = max(1, raw_bytes[hdr_start + 9])
                payload = raw_bytes[offset + 22 : offset + 2 + field_len]

                # Unpack bit-packed 30 subcarriers x (nrx * ntx) complex int8 values
                num_links = min(nrx * ntx, 3)
                num_tones = 30
                expected_vals = num_tones * num_links * 2
                if len(payload) >= expected_vals:
                    iq_int8 = np.frombuffer(
                        payload[:expected_vals], dtype=np.int8
                    ).astype(np.float64)
                    cfr_frame = iq_int8[0::2] + 1j * iq_int8[1::2]
                    packets.append(cfr_frame)
                else:
                    # Bitstream unpack fallback following Halperin's read_bfee.c
                    bit_pos = 0
                    cfr_vals: list[complex] = []
                    payload_len = len(payload)
                    for _ in range(num_tones):
                        bit_pos += 3  # 3-bit subcarrier index header
                        for _ in range(num_links):
                            byte_idx = bit_pos // 8
                            if byte_idx + 2 >= payload_len:
                                break
                            rem = bit_pos % 8
                            raw_r = (payload[byte_idx] >> rem) | (
                                (payload[byte_idx + 1] << (8 - rem)) & 0xFF
                            )
                            raw_i = (payload[byte_idx + 1] >> rem) | (
                                (payload[byte_idx + 2] << (8 - rem)) & 0xFF
                            )
                            real_val = raw_r - 256 if raw_r >= 128 else raw_r
                            imag_val = raw_i - 256 if raw_i >= 128 else raw_i
                            cfr_vals.append(complex(float(real_val), float(imag_val)))
                            bit_pos += 16
                    if cfr_vals:
                        packets.append(np.asarray(cfr_vals, dtype=np.complex128))

            offset += 2 + field_len

        if packets:
            min_k = min(pkt.size for pkt in packets)
            csi_matrix = np.stack([pkt[:min_k] for pkt in packets], axis=1)
            return _slice_or_pad_subcarriers(csi_matrix, target_k=target_k)

        # Fallback: direct binary complex64 array stored in .dat
        usable_bytes = (total_len // 8) * 8
        if usable_bytes == 0:
            raise ValueError(f"Could not decode any valid CSI frames from '{path}'.")
        flat_complex = np.frombuffer(
            raw_bytes[:usable_bytes], dtype=np.complex64
        ).astype(np.complex128)
        if flat_complex.size < target_k:
            flat_complex = np.pad(
                flat_complex, (0, target_k - flat_complex.size), mode="edge"
            )
        num_t = max(1, flat_complex.size // target_k)
        csi_matrix = flat_complex[: target_k * num_t].reshape(target_k, num_t)
        return _slice_or_pad_subcarriers(csi_matrix, target_k=target_k)

    def __len__(self) -> int:
        """Return total number of windowed samples in the dataset."""
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return `(preprocessed_tensor, label)` for window `index`."""
        raw_window = self.windows[index]
        tensor = self.preprocessor.process(raw_window)
        return tensor, int(self.labels[index])


class ESP32CSIDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset for ESP32-CSI-Tool serial `.csv` log files.

    Parses standard `CSI_DATA` CSV rows produced by Espressif's ESP-CSI /
    ESP32-CSI-Tool where the CSI payload column contains a bracketed or
    space/comma-delimited list of interleaved `[imag, real, ...]` values across
    64 FFT bins or 52 active IEEE 802.11n HT20 subcarriers.

    Attributes:
        root_dir: Path to the directory containing `.csv` files.
        num_subcarriers: Number of subcarriers `K` (default: 52).
        window_size: Number of packets `T` per window (default: 200).
        stride: Sliding window stride in packets (default: 100).
        fs: Sampling frequency in Hz (default: 100.0).
        preprocessor: `SignalPreprocessor` instance applied to every window.
        windows: List of extracted raw complex CFR windows `H in C^{K x T}`.
        labels: List of integer labels in `{0, 1, 2, 3, 4}` corresponding to `windows`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_subcarriers: int = 52,
        window_size: int = 200,
        stride: int = 100,
        fs: float = 100.0,
        preprocessor: SignalPreprocessor | None = None,
    ) -> None:
        """Initialize the `ESP32CSIDataset` from a directory of `.csv` logs.

        Args:
            root_dir: Local directory containing ESP32-CSI-Tool `.csv` files.
            num_subcarriers: Target subcarrier count `K` (default: 52).
            window_size: Packets per window `T` (default: 200).
            stride: Packet stride between windows (default: 100).
            fs: Sampling rate in Hz (default: 100.0).
            preprocessor: Optional custom `SignalPreprocessor`.

        Raises:
            FileNotFoundError: If `root_dir` does not exist or contains no `.csv` files,
                with detailed setup instructions for ESP32-CSI-Tool.
        """
        super().__init__()
        self.root_dir: Path = Path(root_dir)
        self.num_subcarriers: int = num_subcarriers
        self.window_size: int = window_size
        self.stride: int = stride
        self.fs: float = float(fs)
        self.preprocessor: SignalPreprocessor = (
            preprocessor
            if preprocessor is not None
            else SignalPreprocessor(num_subcarriers=num_subcarriers, fs=fs)
        )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"ESP32-CSI-Tool dataset directory not found at '{self.root_dir.resolve()}'.\n"
                "Setup Instructions for ESP32-CSI-Tool (.csv):\n"
                "  1. Create the directory: mkdir -p data/raw/esp32\n"
                "  2. Capture IEEE 802.11n 20 MHz CSI packets at 100 Hz using ESP32-CSI-Tool "
                "(https://github.com/StevenMHernandez/ESP32-CSI-Tool) or esp-csi.\n"
                "  3. Save serial `.csv` logs into class subdirectories "
                "(`data/raw/esp32/0_Empty/*.csv` .. `data/raw/esp32/4_Falling/*.csv`) "
                "or prefix filenames with the class ID (`0_empty.csv`, `2_walking.csv`, etc.)."
            )

        csv_files = sorted(self.root_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No ESP32-CSI-Tool '.csv' log files found inside '{self.root_dir.resolve()}'.\n"
                "Setup Instructions: Place ESP32 serial '.csv' CSI logs into "
                f"'{self.root_dir}' (or class subfolders '0_Empty'..'4_Falling') and retry."
            )

        self.windows: list[NDArray[np.complex128]] = []
        self.labels: list[int] = []

        for file_path in csv_files:
            label = _infer_label_from_path(file_path)
            csi_stream = self.parse_csv_file(file_path, target_k=self.num_subcarriers)
            file_windows = _extract_sliding_windows(
                csi_stream,
                window_size=self.window_size,
                stride=self.stride,
            )
            self.windows.extend(file_windows)
            self.labels.extend([label] * len(file_windows))

    @staticmethod
    def parse_csv_file(
        file_path: str | Path,
        target_k: int = 52,
    ) -> NDArray[np.complex128]:
        """Parse an ESP32-CSI-Tool serial `.csv` log file into `H in C^{K x T}`.

        Supports both:
            - Bracketed array columns (e.g. `CSI_DATA,...,\"[i0, r0, i1, r1, ...]\"`
              with 128 values for 64 FFT bins or 104 values for 52 active subcarriers).
            - Pure numeric CSV rows of length `2 * K` (`[real_0, imag_0, ...]`).

        Args:
            file_path: Path to the `.csv` log file.
            target_k: Number of subcarriers `K` to output (default: 52).

        Returns:
            Complex128 NumPy array of shape `(target_k, T_packets)`.

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            ValueError: If no valid CSI packet rows are found in the CSV file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ESP32 CSI .csv file not found: '{path}'.")

        packets: list[NDArray[np.complex128]] = []
        bracket_pattern = re.compile(r"\[([^\]]+)\]")

        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                joined_line = ",".join(row)
                match = bracket_pattern.search(joined_line)
                if match is not None:
                    raw_tokens = re.split(r"[\s,]+", match.group(1).strip())
                    nums = [
                        float(tok)
                        for tok in raw_tokens
                        if tok and re.match(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", tok)
                    ]
                    if len(nums) >= 2 * target_k:
                        arr = np.asarray(nums[: (len(nums) // 2) * 2], dtype=np.float64)
                        # ESP32-CSI-Tool stores [imag, real] pairs per tone
                        cfr = arr[1::2] + 1j * arr[0::2]
                        packets.append(cfr)
                    continue

                # Fallback: numeric CSV row without brackets
                try:
                    nums = [float(cell.strip()) for cell in row if cell.strip()]
                except ValueError:
                    continue
                if len(nums) >= 2 * target_k:
                    arr = np.asarray(nums[: (len(nums) // 2) * 2], dtype=np.float64)
                    cfr = arr[0::2] + 1j * arr[1::2]
                    packets.append(cfr)

        if not packets:
            raise ValueError(
                f"No valid CSI packet rows could be parsed from ESP32 CSV file '{path}'."
            )

        min_k = min(pkt.size for pkt in packets)
        csi_matrix = np.stack([pkt[:min_k] for pkt in packets], axis=1)
        return _slice_or_pad_subcarriers(csi_matrix, target_k=target_k)

    def __len__(self) -> int:
        """Return total number of windowed samples in the dataset."""
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return `(preprocessed_tensor, label)` for window `index`."""
        raw_window = self.windows[index]
        tensor = self.preprocessor.process(raw_window)
        return tensor, int(self.labels[index])
