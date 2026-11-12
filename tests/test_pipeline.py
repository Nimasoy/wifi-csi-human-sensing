"""End-to-end integration tests for dataset generation, augmentations, metrics, and trainer.

Verifies:
1. `SyntheticCSIDataset` deterministic generation across all 5 classes (`0..4`).
2. `CSIAugmenter` training-only augmentations (Gaussian jitter, contiguous subcarrier
   masking up to 5 subcarriers, and contiguous temporal masking up to 15 steps).
3. `compute_classification_metrics` and `compute_presence_fpr` (Binary Presence `1..4`
   vs. `Empty` `0` False Positive Rate).
4. Hardware dataset loader error messages (`Widar3Dataset` & `ESP32CSIDataset`).
5. Single training & validation step via `CSITrainer`.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from src.data.augmentations import CSIAugmenter
from src.data.loaders import ESP32CSIDataset, Widar3Dataset
from src.data.synthetic_csi import SyntheticCSIDataset
from src.engine.metrics import compute_classification_metrics, compute_presence_fpr
from src.engine.trainer import (
    CSITrainer,
    create_stratified_dataloaders,
    set_deterministic_seed,
)
from src.models.csi_net import CSISpatioTemporalNet


class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for data pipeline, augmentations, metrics, and trainer."""

    def setUp(self) -> None:
        """Set deterministic seed before each test."""
        set_deterministic_seed(42)

    def test_synthetic_dataset_determinism_and_shapes(self) -> None:
        """Verify `SyntheticCSIDataset` produces deterministic `(2, 52, 200)` samples."""
        ds_a = SyntheticCSIDataset(num_samples=25, num_subcarriers=52, seed=42)
        ds_b = SyntheticCSIDataset(num_samples=25, num_subcarriers=52, seed=42)

        self.assertEqual(len(ds_a), 25)
        self.assertEqual(set(ds_a.labels.tolist()), {0, 1, 2, 3, 4})

        x_a, y_a = ds_a[3]
        x_b, y_b = ds_b[3]
        self.assertEqual(y_a, y_b)
        self.assertEqual(tuple(x_a.shape), (2, 52, 200))
        torch.testing.assert_close(x_a, x_b)

    def test_csi_augmenter_masks_and_jitter(self) -> None:
        """Verify `CSIAugmenter` applies Gaussian jitter and contiguous masks."""
        augmenter = CSIAugmenter(
            noise_std_range=(0.01, 0.05),
            max_subcarrier_mask=5,
            max_temporal_mask=15,
            p_noise=1.0,
            p_subcarrier_mask=1.0,
            p_temporal_mask=1.0,
            seed=42,
        )
        base_tensor = torch.ones(2, 52, 200, dtype=torch.float32)

        # Test subcarrier mask in isolation
        sub_masked = augmenter.apply_subcarrier_mask(base_tensor)
        zero_subcarriers = int(torch.sum(torch.all(sub_masked[0] == 0.0, dim=-1)).item())
        self.assertGreaterEqual(zero_subcarriers, 1)
        self.assertLessEqual(zero_subcarriers, 5)

        # Test temporal mask in isolation
        time_masked = augmenter.apply_temporal_mask(base_tensor)
        zero_timesteps = int(torch.sum(torch.all(time_masked[0] == 0.0, dim=0)).item())
        self.assertGreaterEqual(zero_timesteps, 1)
        self.assertLessEqual(zero_timesteps, 15)

        # Test full call preserves shape and dtype
        augmented = augmenter(base_tensor)
        self.assertEqual(tuple(augmented.shape), (2, 52, 200))
        self.assertEqual(augmented.dtype, torch.float32)

    def test_metrics_and_presence_fpr_calculation(self) -> None:
        """Verify accuracy, macro-F1, confusion matrix, and binary Presence FPR."""
        # 4 Empty (class 0) samples: 3 predicted Empty (0), 1 predicted Walking (2) -> FPR = 1/4 = 0.25
        y_true = np.array([0, 0, 0, 0, 1, 2, 3, 4], dtype=np.int64)
        y_pred = np.array([0, 0, 0, 2, 1, 2, 3, 4], dtype=np.int64)

        fpr = compute_presence_fpr(y_true, y_pred)
        self.assertAlmostEqual(fpr, 0.25, places=6)

        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics.accuracy, 7.0 / 8.0, places=6)
        self.assertAlmostEqual(metrics.presence_fpr, 0.25, places=6)
        self.assertEqual(metrics.confusion_matrix.shape, (5, 5))
        self.assertEqual(int(metrics.confusion_matrix[0, 0]), 3)
        self.assertEqual(int(metrics.confusion_matrix[0, 2]), 1)

    def test_hardware_loaders_raise_informative_filenotfounderror(self) -> None:
        """Verify `Widar3Dataset` and `ESP32CSIDataset` raise clear setup instructions."""
        with self.assertRaises(FileNotFoundError) as widar_ctx:
            Widar3Dataset(root_dir="non_existent_widar3_dir_xyz")
        self.assertIn("Setup Instructions", str(widar_ctx.exception))

        with self.assertRaises(FileNotFoundError) as esp_ctx:
            ESP32CSIDataset(root_dir="non_existent_esp32_dir_xyz")
        self.assertIn("Setup Instructions", str(esp_ctx.exception))

    def test_single_training_and_validation_step(self) -> None:
        """Verify stratified splitting and a 1-epoch train/val/test step via `CSITrainer`."""
        dataset = SyntheticCSIDataset(
            num_samples=40,
            num_subcarriers=52,
            window_size=200,
            seed=42,
            precompute=True,
        )
        train_loader, val_loader, test_loader = create_stratified_dataloaders(
            dataset=dataset,
            labels=dataset.labels,
            batch_size=16,
            seed=42,
            augment_train=True,
        )

        model = CSISpatioTemporalNet()
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = CSITrainer(
                model=model,
                output_dir=tmp_dir,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=1e-4,
            )
            report = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            self.assertEqual(report["best_epoch"], 1)
            self.assertIn("test_evaluation", report)
            self.assertIsNotNone(report["test_evaluation"])


if __name__ == "__main__":
    unittest.main()
