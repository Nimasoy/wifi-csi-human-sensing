"""CLI training script for the Wi-Fi CSI Human Presence & Motion Sensing System.

Usage:
    # Full 50-epoch training using default configuration:
    python scripts/train.py --config configs/default_config.yaml

    # Fast 2-epoch smoke test on a compact synthetic dataset:
    python scripts/train.py --config configs/default_config.yaml --smoke-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

# Ensure repository root is importable when executed as `python scripts/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import ESP32CSIDataset, Widar3Dataset
from src.data.synthetic_csi import SyntheticCSIDataset
from src.dsp.preprocessor import SignalPreprocessor
from src.engine.trainer import (
    CSITrainer,
    create_stratified_dataloaders,
    set_deterministic_seed,
)
from src.models.csi_net import CSISpatioTemporalNet


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: '{path.resolve()}'.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return dict(data) if isinstance(data, dict) else {}


def build_preprocessor_from_config(cfg: dict[str, Any]) -> SignalPreprocessor:
    """Instantiate `SignalPreprocessor` matching `configs/default_config.yaml`."""
    sig_cfg = cfg.get("signal", {})
    dsp_cfg = cfg.get("dsp", {})
    hampel_cfg = dsp_cfg.get("hampel", {})
    butter_cfg = dsp_cfg.get("butterworth", {})
    stft_cfg = dsp_cfg.get("stft", {})
    norm_cfg = dsp_cfg.get("normalization", {})

    return SignalPreprocessor(
        num_subcarriers=int(sig_cfg.get("num_subcarriers", 52)),
        fs=float(sig_cfg.get("sampling_rate_hz", 100.0)),
        hampel_window_size=int(hampel_cfg.get("window_size", 7)),
        hampel_n_sigmas=float(hampel_cfg.get("n_sigmas", 3.0)),
        hampel_mad_scale=float(hampel_cfg.get("mad_scale", 1.4826)),
        lowcut_hz=float(butter_cfg.get("lowcut_hz", 0.1)),
        highcut_hz=float(butter_cfg.get("highcut_hz", 15.0)),
        filter_order=int(butter_cfg.get("order", 3)),
        stft_n_fft=int(stft_cfg.get("n_fft", 64)),
        stft_win_length=int(stft_cfg.get("win_length", 32)),
        stft_hop_length=int(stft_cfg.get("hop_length", 8)),
        norm_eps=float(norm_cfg.get("eps", 1e-8)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for `scripts/train.py`."""
    parser = argparse.ArgumentParser(
        description="Train CSISpatioTemporalNet for Wi-Fi CSI Human Presence & Motion Sensing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file (default: configs/default_config.yaml).",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["synthetic", "esp32", "widar3"],
        default="synthetic",
        help="Dataset backend to train on (default: synthetic).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root directory for hardware datasets when --dataset-type is esp32 or widar3.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints and evaluation JSON reports.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2-epoch smoke test on a compact synthetic dataset (100 samples).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute end-to-end training pipeline and return the diagnostic report."""
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_deterministic_seed(seed)

    sig_cfg = cfg.get("signal", {})
    syn_cfg = cfg.get("synthetic_data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    num_classes = int(sig_cfg.get("num_classes", 5))
    num_subcarriers = int(sig_cfg.get("num_subcarriers", 52))
    window_size = int(sig_cfg.get("window_size_packets", 200))
    stride = int(sig_cfg.get("stride_packets", 100))
    fs = float(sig_cfg.get("sampling_rate_hz", 100.0))

    if args.dataset_type == "widar3" and "num_subcarriers" not in sig_cfg:
        num_subcarriers = 90

    preprocessor = build_preprocessor_from_config(cfg)

    if args.smoke_test:
        epochs = 2
        batch_size = 16
        total_samples = 100  # 20 samples per class for fast 70/15/15 stratified split
    else:
        epochs = (
            int(args.epochs)
            if args.epochs is not None
            else int(train_cfg.get("epochs", 50))
        )
        batch_size = int(train_cfg.get("batch_size", 32))
        samples_per_class = int(syn_cfg.get("num_samples_per_class", 200))
        total_samples = samples_per_class * num_classes

    if args.dataset_type == "synthetic":
        dataset = SyntheticCSIDataset(
            num_samples=total_samples,
            num_subcarriers=num_subcarriers,
            window_size=window_size,
            stride=stride,
            fs=fs,
            seed=seed,
            preprocessor=preprocessor,
            precompute=True,
        )
        labels = dataset.labels
    elif args.dataset_type == "esp32":
        data_root = args.data_dir if args.data_dir is not None else "data/raw/esp32"
        dataset = ESP32CSIDataset(
            root_dir=data_root,
            num_subcarriers=num_subcarriers,
            window_size=window_size,
            stride=stride,
            fs=fs,
            preprocessor=preprocessor,
        )
        labels = dataset.labels
    else:
        data_root = args.data_dir if args.data_dir is not None else "data/raw/widar3"
        dataset = Widar3Dataset(
            root_dir=data_root,
            num_subcarriers=num_subcarriers,
            window_size=window_size,
            stride=stride,
            fs=fs,
            preprocessor=preprocessor,
        )
        labels = dataset.labels

    train_loader, val_loader, test_loader = create_stratified_dataloaders(
        dataset=dataset,
        labels=labels,
        batch_size=batch_size,
        train_split=float(train_cfg.get("train_split", 0.70)),
        val_split=float(train_cfg.get("val_split", 0.15)),
        test_split=float(train_cfg.get("test_split", 0.15)),
        seed=seed,
        augment_train=True,
    )

    model = CSISpatioTemporalNet(
        in_channels=int(model_cfg.get("in_channels", 2)),
        num_classes=num_classes,
        temporal_pooled_len=int(model_cfg.get("temporal_pooled_len", 100)),
        lstm_hidden_size=int(model_cfg.get("lstm_hidden_size", 128)),
        lstm_num_layers=int(model_cfg.get("lstm_num_layers", 2)),
        lstm_dropout=float(model_cfg.get("lstm_dropout", 0.30)),
        attention_dim=int(model_cfg.get("attention_dim", 128)),
    )

    trainer = CSITrainer(
        model=model,
        output_dir=args.output_dir,
        epochs=epochs,
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        label_smoothing=0.1,
        max_grad_norm=1.0,
    )

    report = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        checkpoint_name="best_model.pt",
        report_name="evaluation_summary.json",
        extra_metadata={
            "seed": seed,
            "smoke_test": bool(args.smoke_test),
            "dataset_type": args.dataset_type,
            "num_subcarriers": num_subcarriers,
            "window_size": window_size,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )
    return report


if __name__ == "__main__":
    main()
