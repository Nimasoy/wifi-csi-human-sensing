"""Training, evaluation, learning-rate scheduling, and metric computation engine."""

from src.engine.metrics import (
    EvaluationMetrics,
    compute_classification_metrics,
    compute_presence_fpr,
)
from src.engine.trainer import (
    CSITrainer,
    build_warmup_cosine_scheduler,
    create_stratified_dataloaders,
    set_deterministic_seed,
)

__all__ = [
    "CSITrainer",
    "EvaluationMetrics",
    "build_warmup_cosine_scheduler",
    "compute_classification_metrics",
    "compute_presence_fpr",
    "create_stratified_dataloaders",
    "set_deterministic_seed",
]
