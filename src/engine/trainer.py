"""Training and evaluation engine for the Wi-Fi CSI Spatio-Temporal Network.

Specifications implemented:
- Loss: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- Optimizer: `torch.optim.AdamW(lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)`
- Gradient Clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Learning Rate Scheduler:
  5-epoch linear warmup (`1e-5` -> `1e-3`) followed by
  `CosineAnnealingLR(T_max=max(1, epochs - 5), eta_min=1e-6)`
  (handles `epochs <= 5` safely for smoke tests).
- Data Splitting: Stratified `70% / 15% / 15%` Train / Val / Test split
  (`batch_size = 32`, `epochs = 50`, `seed = 42`), applying `CSIAugmenter`
  exclusively to the training split.
- Checkpointing & Diagnostics: Saves best model checkpoint by validation
  Macro-F1 (`best_model.pt`) and exports a structured JSON diagnostic report.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.augmentations import AugmentedCSIDataset, CSIAugmenter
from src.data.synthetic_csi import CLASS_LABELS
from src.engine.metrics import EvaluationMetrics, compute_classification_metrics


def set_deterministic_seed(seed: int = 42) -> None:
    """Seed Python `random`, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Args:
        seed: Deterministic integer seed (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_stratified_dataloaders(
    dataset: Dataset[tuple[torch.Tensor, int]],
    labels: NDArray[np.int64] | list[int],
    batch_size: int = 32,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    augment_train: bool = True,
    num_workers: int = 0,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
]:
    """Create stratified 70% / 15% / 15% Train, Validation, and Test DataLoaders.

    Applies `CSIAugmenter` strictly on the training split while keeping validation
    and test splits completely unaugmented.

    Args:
        dataset: Full dataset yielding `(preprocessed_tensor, label)`.
        labels: Sequence or array of integer labels for all samples in `dataset`.
        batch_size: Mini-batch size (default: 32).
        train_split: Proportion of data allocated to training (default: 0.70).
        val_split: Proportion of data allocated to validation (default: 0.15).
        test_split: Proportion of data allocated to testing (default: 0.15).
        seed: Deterministic random seed (default: 42).
        augment_train: Whether to wrap the training split with `CSIAugmenter`.
        num_workers: Number of DataLoader worker processes (default: 0).

    Returns:
        Tuple `(train_loader, val_loader, test_loader)`.

    Raises:
        ValueError: If split ratios do not sum to 1.0 within `1e-5`.
    """
    total_ratio = train_split + val_split + test_split
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"train_split + val_split + test_split must equal 1.0, got {total_ratio:.4f}."
        )

    labels_arr = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(len(labels_arr), dtype=np.int64)

    holdout_ratio = val_split + test_split
    train_idx, temp_idx, _, temp_labels = train_test_split(
        all_indices,
        labels_arr,
        test_size=holdout_ratio,
        stratify=labels_arr,
        random_state=seed,
    )

    relative_test_ratio = test_split / holdout_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    train_subset: Dataset[tuple[torch.Tensor, int]] = Subset(
        dataset, train_idx.tolist()
    )
    val_subset: Dataset[tuple[torch.Tensor, int]] = Subset(dataset, val_idx.tolist())
    test_subset: Dataset[tuple[torch.Tensor, int]] = Subset(dataset, test_idx.tolist())

    augmenter = CSIAugmenter(seed=seed) if augment_train else None
    train_dataset = AugmentedCSIDataset(train_subset, augmenter=augmenter)
    val_dataset = AugmentedCSIDataset(val_subset, augmenter=None)
    test_dataset = AugmentedCSIDataset(test_subset, augmenter=None)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    warmup_epochs: int = 5,
    base_lr: float = 1e-3,
    warmup_start_lr: float = 1e-5,
    eta_min: float = 1e-6,
) -> LRScheduler:
    """Build 5-epoch linear warmup (`1e-5` -> `1e-3`) + CosineAnnealingLR scheduler.

    Handles `epochs <= warmup_epochs` safely (e.g., 2-epoch smoke tests) without
    zero-division or negative `T_max`.

    Args:
        optimizer: Configured PyTorch optimizer (`AdamW` with initial `lr = base_lr`).
        epochs: Total number of training epochs (default: 50).
        warmup_epochs: Number of linear warmup epochs (default: 5).
        base_lr: Target peak learning rate after warmup (default: `1e-3`).
        warmup_start_lr: Initial learning rate at epoch 0 (default: `1e-5`).
        eta_min: Minimum learning rate for cosine annealing (default: `1e-6`).

    Returns:
        Configured PyTorch `LRScheduler`.
    """
    start_factor = warmup_start_lr / base_lr

    if epochs <= warmup_epochs:
        effective_steps = max(1, epochs - 1)

        def short_warmup_lambda(epoch_idx: int) -> float:
            progress = min(1.0, float(epoch_idx) / float(effective_steps))
            return start_factor + (1.0 - start_factor) * progress

        return LambdaLR(optimizer, lr_lambda=short_warmup_lambda)

    def warmup_lambda(epoch_idx: int) -> float:
        progress = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
        return start_factor + (1.0 - start_factor) * progress

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


class CSITrainer:
    """End-to-end training and evaluation engine for `CSISpatioTemporalNet`.

    Attributes:
        model: `CSISpatioTemporalNet` instance.
        device: Target `torch.device` (`cuda` or `cpu`).
        epochs: Total training epochs (default: 50).
        learning_rate: Peak learning rate (default: `1e-3`).
        weight_decay: AdamW weight decay (default: `1e-4`).
        label_smoothing: Cross-entropy label smoothing factor (default: `0.1`).
        max_grad_norm: Maximum gradient L2 norm for clipping (default: `1.0`).
        output_dir: Directory where checkpoints and JSON diagnostics are saved.
        criterion: `nn.CrossEntropyLoss(label_smoothing=0.1)`.
        optimizer: `AdamW(lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)`.
        scheduler: 5-epoch linear warmup (`1e-5` -> `1e-3`) + `CosineAnnealingLR`.
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str | Path = "outputs",
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-5,
        eta_min: float = 1e-6,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the `CSITrainer` engine.

        Args:
            model: Neural network module (`CSISpatioTemporalNet`).
            output_dir: Output directory for saving checkpoints and JSON summaries.
            epochs: Total number of training epochs (default: 50).
            learning_rate: Target learning rate after warmup (default: `1e-3`).
            weight_decay: AdamW weight decay (default: `1e-4`).
            label_smoothing: Label smoothing factor for CrossEntropyLoss (default: `0.1`).
            max_grad_norm: Max gradient norm for `clip_grad_norm_` (default: `1.0`).
            warmup_epochs: Number of linear warmup epochs (default: 5).
            warmup_start_lr: Starting LR at epoch 1 (default: `1e-5`).
            eta_min: Final minimum LR for cosine annealing (default: `1e-6`).
            device: Optional compute device (`cpu` or `cuda`). Auto-detected if `None`.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: nn.Module = model.to(self.device)
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs: int = int(epochs)
        self.learning_rate: float = float(learning_rate)
        self.weight_decay: float = float(weight_decay)
        self.label_smoothing: float = float(label_smoothing)
        self.max_grad_norm: float = float(max_grad_norm)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        self.scheduler: LRScheduler = build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            epochs=self.epochs,
            warmup_epochs=warmup_epochs,
            base_lr=self.learning_rate,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min,
        )

    def train_one_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
    ) -> tuple[float, EvaluationMetrics]:
        """Run one full training epoch with gradient clipping.

        Args:
            train_loader: Training `DataLoader`.

        Returns:
            Tuple `(avg_train_loss, train_metrics)`.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = self.criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()

            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

        avg_loss = total_loss / max(1, total_samples)
        metrics = compute_classification_metrics(all_targets, all_preds)
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader[tuple[torch.Tensor, int]],
    ) -> tuple[float, EvaluationMetrics]:
        """Evaluate the model on a validation or test `DataLoader`.

        Args:
            data_loader: Validation or test `DataLoader` (unaugmented).

        Returns:
            Tuple `(avg_loss, eval_metrics)`.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = self.criterion(logits, targets)
            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

        avg_loss = total_loss / max(1, total_samples)
        metrics = compute_classification_metrics(all_targets, all_preds)
        return avg_loss, metrics

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        epoch: int,
        val_metrics: EvaluationMetrics,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model state dict, optimizer state, and validation metrics to disk.

        Args:
            checkpoint_path: Destination `.pt` file path.
            epoch: 1-based epoch number.
            val_metrics: Validation `EvaluationMetrics` at this checkpoint.
            extra_metadata: Optional extra configuration metadata to embed.

        Returns:
            Resolved `Path` to the saved checkpoint.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_macro_f1": val_metrics.macro_f1,
            "val_accuracy": val_metrics.accuracy,
            "val_metrics": val_metrics.to_dict(),
            "class_labels": CLASS_LABELS,
        }
        if extra_metadata is not None:
            payload["metadata"] = extra_metadata

        torch.save(payload, path)
        return path

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
        val_loader: DataLoader[tuple[torch.Tensor, int]],
        test_loader: DataLoader[tuple[torch.Tensor, int]] | None = None,
        checkpoint_name: str = "best_model.pt",
        report_name: str = "evaluation_summary.json",
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run full training loop, save best Macro-F1 checkpoint, and export JSON report.

        Args:
            train_loader: Training `DataLoader` (with `CSIAugmenter`).
            val_loader: Validation `DataLoader` (unaugmented).
            test_loader: Optional Test `DataLoader` evaluated using the best checkpoint.
            checkpoint_name: Filename for the best validation Macro-F1 checkpoint.
            report_name: Filename for the diagnostic JSON report.
            extra_metadata: Optional configuration dictionary saved in the report.

        Returns:
            Dictionary containing training history, best validation metrics,
            test metrics (if `test_loader` is provided), and output file paths.
        """
        best_val_f1 = -1.0
        best_epoch = 0
        best_val_metrics: EvaluationMetrics | None = None
        checkpoint_path = self.output_dir / checkpoint_name
        history: list[dict[str, Any]] = []
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            train_loss, train_metrics = self.train_one_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_metrics.accuracy,
                "train_macro_f1": train_metrics.macro_f1,
                "val_loss": val_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
                "val_presence_fpr": val_metrics.presence_fpr,
            }
            history.append(epoch_record)

            if val_metrics.macro_f1 >= best_val_f1:
                best_val_f1 = val_metrics.macro_f1
                best_epoch = epoch
                best_val_metrics = val_metrics
                self.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    extra_metadata=extra_metadata,
                )

            print(
                f"[Epoch {epoch:02d}/{self.epochs:02d}] "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} (Acc: {train_metrics.accuracy:.4f}, "
                f"F1: {train_metrics.macro_f1:.4f}) | "
                f"Val Loss: {val_loss:.4f} (Acc: {val_metrics.accuracy:.4f}, "
                f"F1: {val_metrics.macro_f1:.4f}, FPR: {val_metrics.presence_fpr:.4f})"
            )

        elapsed_sec = time.time() - start_time
        test_summary: dict[str, Any] | None = None

        if test_loader is not None and checkpoint_path.exists():
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            test_loss, test_metrics = self.evaluate(test_loader)
            test_summary = {
                "test_loss": test_loss,
                **test_metrics.to_dict(),
            }
            print(
                f"[Test Split Evaluation @ Best Epoch {best_epoch}] "
                f"Acc: {test_metrics.accuracy:.4f} | "
                f"Macro-F1: {test_metrics.macro_f1:.4f} | "
                f"Presence FPR: {test_metrics.presence_fpr:.4f}"
            )

        diagnostic_report: dict[str, Any] = {
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_f1),
            "best_val_metrics": (
                best_val_metrics.to_dict() if best_val_metrics is not None else None
            ),
            "test_evaluation": test_summary,
            "training_time_sec": round(elapsed_sec, 3),
            "checkpoint_path": str(checkpoint_path),
            "history": history,
        }
        if extra_metadata is not None:
            diagnostic_report["config"] = extra_metadata

        report_path = self.output_dir / report_name
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(diagnostic_report, handle, indent=2)

        return diagnostic_report
