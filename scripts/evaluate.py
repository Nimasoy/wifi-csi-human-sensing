"""CLI evaluation script for the Wi-Fi CSI Human Presence & Motion Sensing System.

Loads a trained `CSISpatioTemporalNet` checkpoint (`outputs/best_model.pt`),
evaluates on the held-out stratified 15% test split (or a hardware dataset),
prints comprehensive metrics (Top-1 Accuracy, Macro Precision/Recall/F1,
Presence vs. Empty FPR, and 5x5 Confusion Matrix), and saves both a confusion
matrix JSON/CSV and diagnostic report to `outputs/`.

Usage:
    python scripts/evaluate.py --config configs/default_config.yaml --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_preprocessor_from_config, load_yaml_config
from src.data.loaders import ESP32CSIDataset, Widar3Dataset
from src.data.synthetic_csi import CLASS_LABELS, SyntheticCSIDataset
from src.engine.trainer import (
    CSITrainer,
    create_stratified_dataloaders,
    set_deterministic_seed,
)
from src.models.csi_net import CSISpatioTemporalNet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for `scripts/evaluate.py`."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CSISpatioTemporalNet checkpoint on the test split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file (default: configs/default_config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_model.pt",
        help="Path to saved model checkpoint (.pt) file.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["synthetic", "esp32", "widar3"],
        default="synthetic",
        help="Dataset backend to evaluate on (default: synthetic).",
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
        help="Directory to write confusion matrix and evaluation diagnostic JSON.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate on the compact 100-sample synthetic dataset split used in smoke-test.",
    )
    return parser.parse_args(argv)


def format_confusion_matrix_table(conf_mat: np.ndarray) -> str:
    """Format the 5x5 confusion matrix as a human-readable ASCII table."""
    header_names = [CLASS_LABELS[i] for i in range(conf_mat.shape[0])]
    col_width = max(len(name) for name in header_names) + 2
    header_line = "True \\ Pred".ljust(col_width) + "".join(
        name.rjust(col_width) for name in header_names
    )
    lines = [header_line, "-" * len(header_line)]
    for row_idx, row in enumerate(conf_mat):
        row_str = CLASS_LABELS[row_idx].ljust(col_width) + "".join(
            f"{int(val)}".rjust(col_width) for val in row
        )
        lines.append(row_str)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Load checkpoint, evaluate on the test split, and save diagnostic artifacts."""
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{ckpt_path.resolve()}'. "
            "Run `python scripts/train.py --config configs/default_config.yaml` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_meta = checkpoint.get("metadata", {})

    seed = int(ckpt_meta.get("seed", cfg.get("project", {}).get("seed", 42)))
    set_deterministic_seed(seed)

    sig_cfg = cfg.get("signal", {})
    syn_cfg = cfg.get("synthetic_data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    num_classes = int(sig_cfg.get("num_classes", 5))
    num_subcarriers = int(
        ckpt_meta.get("num_subcarriers", sig_cfg.get("num_subcarriers", 52))
    )
    window_size = int(
        ckpt_meta.get("window_size", sig_cfg.get("window_size_packets", 200))
    )
    stride = int(sig_cfg.get("stride_packets", 100))
    fs = float(sig_cfg.get("sampling_rate_hz", 100.0))

    is_smoke = bool(args.smoke_test or ckpt_meta.get("smoke_test", False))
    batch_size = int(ckpt_meta.get("batch_size", train_cfg.get("batch_size", 32)))
    total_samples = (
        100
        if is_smoke
        else int(syn_cfg.get("num_samples_per_class", 200)) * num_classes
    )

    preprocessor = build_preprocessor_from_config(cfg)

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

    _, _, test_loader = create_stratified_dataloaders(
        dataset=dataset,
        labels=labels,
        batch_size=batch_size,
        train_split=float(train_cfg.get("train_split", 0.70)),
        val_split=float(train_cfg.get("val_split", 0.15)),
        test_split=float(train_cfg.get("test_split", 0.15)),
        seed=seed,
        augment_train=False,
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
    model.load_state_dict(checkpoint["model_state_dict"])

    trainer = CSITrainer(
        model=model,
        output_dir=args.output_dir,
        epochs=1,
        device=device,
    )
    test_loss, test_metrics = trainer.evaluate(test_loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix CSV and diagnostic JSON
    conf_mat_csv_path = output_dir / "confusion_matrix.csv"
    np.savetxt(
        conf_mat_csv_path,
        test_metrics.confusion_matrix,
        fmt="%d",
        delimiter=",",
        header=",".join(CLASS_LABELS[i] for i in range(num_classes)),
        comments="",
    )

    diagnostic_payload: dict[str, Any] = {
        "checkpoint_path": str(ckpt_path),
        "trained_epoch": int(checkpoint.get("epoch", 0)),
        "test_loss": float(test_loss),
        **test_metrics.to_dict(),
    }
    eval_json_path = output_dir / "test_diagnostics.json"
    with eval_json_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic_payload, handle, indent=2)

    print("=" * 78)
    print("Wi-Fi CSI Human Presence & Motion Sensing — Test Split Evaluation")
    print("=" * 78)
    print(f"Checkpoint            : {ckpt_path} (Epoch {checkpoint.get('epoch', '?')})")
    print(f"Test Loss             : {test_loss:.4f}")
    print(f"Top-1 Accuracy        : {test_metrics.accuracy * 100:.2f}%")
    print(f"Macro Precision       : {test_metrics.macro_precision * 100:.2f}%")
    print(f"Macro Recall          : {test_metrics.macro_recall * 100:.2f}%")
    print(f"Macro F1-Score        : {test_metrics.macro_f1 * 100:.2f}%")
    print(f"Presence vs Empty FPR : {test_metrics.presence_fpr * 100:.2f}%")
    print("-" * 78)
    print("5x5 Confusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    print(format_confusion_matrix_table(test_metrics.confusion_matrix))
    print("=" * 78)

    return diagnostic_payload


if __name__ == "__main__":
    main()
