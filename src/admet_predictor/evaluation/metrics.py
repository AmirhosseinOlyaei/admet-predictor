"""Evaluation metrics for ADMET prediction."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_auroc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute Area Under the ROC Curve.

    Returns 0.5 if there is only one class in y_true.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred_proba))


def compute_auprc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_pred_proba))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(mean_absolute_error(y_true, y_pred))


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_composite_score(
    per_task_metrics: dict[str, float],
    task_configs: list[dict],
) -> float:
    """Compute weighted geometric mean composite score.

    Normalization per metric type:
    - AUROC:    score = AUROC (already in [0, 1])
    - MAE:      score = 1 / (1 + MAE)
    - Spearman: score = (rho + 1) / 2

    Parameters
    ----------
    per_task_metrics:
        Dict mapping task_name → raw metric value.
    task_configs:
        List of task config dicts with 'metric' and 'weight' keys.

    Returns
    -------
    Weighted geometric mean in [0, 1].
    """
    scores: list[float] = []
    weights: list[float] = []

    for tc in task_configs:
        name = tc["name"]
        if name not in per_task_metrics:
            continue

        raw = per_task_metrics[name]
        metric = tc.get("metric", "MAE")

        if metric == "AUROC":
            score = float(np.clip(raw, 0.0, 1.0))
        elif metric == "MAE":
            score = 1.0 / (1.0 + max(float(raw), 0.0))
        elif metric == "Spearman":
            score = (float(raw) + 1.0) / 2.0
        else:
            score = float(raw)

        scores.append(max(score, 1e-8))
        weights.append(tc.get("weight", 1.0))

    if not scores:
        return 0.0

    log_sum = sum(w * np.log(s) for s, w in zip(scores, weights))
    return float(np.exp(log_sum / sum(weights)))
