"""Evaluation metrics for 5-class Wi-Fi CSI human presence & motion sensing.

Computes:
1. Overall Top-1 Accuracy.
2. Macro-Averaged Precision, Recall, and F1-Score (plus per-class metrics).
3. 5x5 Confusion Matrix.
4. Binary Presence (classes `1..4`) vs. `Empty` (class `0`) False Positive Rate (FPR):
       FPR = FP_presence / (FP_presence + TN_empty)
   where `Empty` (`0`) is the negative class and `Presence` (`1, 2, 3, 4`) is the
   positive class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.synthetic_csi import CLASS_LABELS


@dataclass(frozen=True)
class EvaluationMetrics:
    """Structured container for multi-class and binary presence evaluation metrics.

    Attributes:
        accuracy: Overall Top-1 classification accuracy in `[0.0, 1.0]`.
        macro_precision: Macro-averaged precision across all 5 classes.
        macro_recall: Macro-averaged recall across all 5 classes.
        macro_f1: Macro-averaged F1-score across all 5 classes.
        presence_fpr: Binary False Positive Rate for `Presence (1..4)` vs. `Empty (0)`.
        confusion_matrix: 2D integer array of shape `(5, 5)` where rows are true
            labels and columns are predicted labels.
        per_class_metrics: Dictionary mapping each class name to its precision,
            recall, f1, and support.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    presence_fpr: float
    confusion_matrix: NDArray[np.int64]
    per_class_metrics: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all evaluation metrics to a JSON-compatible dictionary."""
        return {
            "accuracy": float(self.accuracy),
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "presence_fpr": float(self.presence_fpr),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_metrics": self.per_class_metrics,
        }


def compute_presence_fpr(
    y_true: NDArray[np.integer] | list[int],
    y_pred: NDArray[np.integer] | list[int],
    empty_class_id: int = 0,
) -> float:
    """Compute binary False Positive Rate (FPR) for Presence (`1..4`) vs. Empty (`0`).

    Defines:
        - Negative class (`0`): `y == empty_class_id` (`Empty`)
        - Positive class (`1`): `y != empty_class_id` (`Stationary_Presence`,
          `Walking`, `Sitting_Standing`, `Falling`)
        - `FP`: True label is `Empty (0)`, predicted label is in `{1, 2, 3, 4}`.
        - `TN`: True label is `Empty (0)`, predicted label is `Empty (0)`.
        - `FPR = FP / (FP + TN)` (returns `0.0` if there are no `Empty` ground-truth
          samples in the evaluated split).

    Args:
        y_true: Ground-truth class indices in `{0, 1, 2, 3, 4}`.
        y_pred: Predicted class indices in `{0, 1, 2, 3, 4}`.
        empty_class_id: Integer ID corresponding to the `Empty` class (default: 0).

    Returns:
        Binary presence False Positive Rate in `[0.0, 1.0]`.
    """
    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)

    empty_mask = true_arr == empty_class_id
    total_empty = int(np.sum(empty_mask))
    if total_empty == 0:
        return 0.0

    false_positives = int(np.sum(empty_mask & (pred_arr != empty_class_id)))
    return float(false_positives / total_empty)


def compute_classification_metrics(
    y_true: NDArray[np.integer] | list[int],
    y_pred: NDArray[np.integer] | list[int],
    num_classes: int = 5,
    class_labels: dict[int, str] | None = None,
) -> EvaluationMetrics:
    """Compute all 5-class classification metrics and binary Presence vs. Empty FPR.

    Args:
        y_true: Ground-truth integer labels of length `N`.
        y_pred: Predicted integer labels of length `N`.
        num_classes: Total number of classes `C` (default: 5).
        class_labels: Optional mapping `{class_id: class_name}`. Defaults to
            `CLASS_LABELS`.

    Returns:
        Populated `EvaluationMetrics` instance.

    Raises:
        ValueError: If `y_true` and `y_pred` are empty or have mismatched lengths.
    """
    true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()

    if true_arr.size == 0 or true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must be non-empty 1D arrays of equal length, "
            f"got shapes {true_arr.shape} and {pred_arr.shape}."
        )

    labels_map = class_labels if class_labels is not None else CLASS_LABELS
    label_indices = list(range(num_classes))

    acc = float(accuracy_score(true_arr, pred_arr))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    conf_mat = confusion_matrix(
        true_arr,
        pred_arr,
        labels=label_indices,
    ).astype(np.int64)
    fpr = compute_presence_fpr(true_arr, pred_arr, empty_class_id=0)

    per_class_dict: dict[str, dict[str, float | int]] = {}
    for idx in label_indices:
        cname = labels_map.get(idx, f"Class_{idx}")
        per_class_dict[cname] = {
            "precision": float(np.asarray(per_p)[idx]),
            "recall": float(np.asarray(per_r)[idx]),
            "f1": float(np.asarray(per_f1)[idx]),
            "support": int(np.asarray(per_sup)[idx]),
        }

    return EvaluationMetrics(
        accuracy=acc,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        presence_fpr=fpr,
        confusion_matrix=conf_mat,
        per_class_metrics=per_class_dict,
    )
