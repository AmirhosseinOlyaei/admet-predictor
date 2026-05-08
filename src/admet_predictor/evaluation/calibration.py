"""Calibration utilities for classification probability outputs."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import Tensor


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true:
        Binary ground-truth labels, shape [N].
    y_prob:
        Predicted probabilities, shape [N].
    n_bins:
        Number of equal-width confidence bins.

    Returns
    -------
    ECE as a float in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        if not mask.any():
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_count = mask.sum()
        ece += (bin_count / n) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for a reliability (calibration) diagram.

    Returns
    -------
    Dict with keys:
    - bin_confidences: list of mean predicted probabilities per bin
    - bin_accuracies:  list of mean observed accuracies per bin
    - bin_counts:      list of sample counts per bin
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        count = int(mask.sum())
        if count == 0:
            bin_confidences.append(float((low + high) / 2))
            bin_accuracies.append(0.0)
            bin_counts.append(0)
        else:
            bin_confidences.append(float(y_prob[mask].mean()))
            bin_accuracies.append(float(y_true[mask].mean()))
            bin_counts.append(count)

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def temperature_scale(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature scaling to logits.

    Parameters
    ----------
    logits:
        Raw model logits, shape [N] or [N, 1].
    temperature:
        Temperature value T > 0. T=1 is no scaling.

    Returns
    -------
    Scaled logits of the same shape.
    """
    return logits / max(float(temperature), 1e-6)


def _nll_loss(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary cross-entropy NLL at a given temperature."""
    scaled = logits / max(temperature, 1e-6)
    # Numerically stable sigmoid
    probs = np.where(
        scaled >= 0,
        1.0 / (1.0 + np.exp(-scaled)),
        np.exp(scaled) / (1.0 + np.exp(scaled)),
    )
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(nll)


def find_optimal_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 10.0,
) -> float:
    """Find the temperature that minimises NLL on a validation set.

    Uses scipy's bounded scalar minimiser (golden section / Brent's method).

    Parameters
    ----------
    val_logits:
        Validation logits, shape [N].
    val_labels:
        Binary validation labels, shape [N].
    t_min, t_max:
        Search bounds for temperature.

    Returns
    -------
    Optimal temperature as a float.
    """
    val_logits = np.asarray(val_logits, dtype=np.float64).ravel()
    val_labels = np.asarray(val_labels, dtype=np.float64).ravel()

    result = minimize_scalar(
        lambda t: _nll_loss(t, val_logits, val_labels),
        bounds=(t_min, t_max),
        method="bounded",
    )
    return float(result.x)
