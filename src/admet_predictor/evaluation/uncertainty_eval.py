"""Uncertainty quantification evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_ence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_uncertainty: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Normalized Calibration Error (ENCE) for regression.

    The ENCE measures how well predicted uncertainties match observed errors.
    Bins molecules by predicted uncertainty; within each bin, checks that
    the mean uncertainty equals the RMSE.

    Parameters
    ----------
    y_true:
        Ground-truth regression targets, shape [N].
    y_pred:
        Predicted values, shape [N].
    y_uncertainty:
        Predicted standard deviations (sqrt of variance), shape [N].
    n_bins:
        Number of equal-width bins over the uncertainty range.

    Returns
    -------
    ENCE (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_uncertainty = np.asarray(y_uncertainty, dtype=np.float64)

    errors = np.abs(y_true - y_pred)

    # Bin by predicted uncertainty
    unc_min, unc_max = y_uncertainty.min(), y_uncertainty.max()
    if unc_max == unc_min:
        return 0.0

    bin_edges = np.linspace(unc_min, unc_max, n_bins + 1)
    ence = 0.0
    count_total = 0

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_uncertainty >= low) & (y_uncertainty <= high)
        if not mask.any():
            continue
        mean_unc = y_uncertainty[mask].mean()
        rmse = np.sqrt(np.mean(errors[mask] ** 2))
        bin_n = mask.sum()
        ence += bin_n * abs(mean_unc - rmse) / max(mean_unc, 1e-8)
        count_total += bin_n

    return float(ence / max(count_total, 1))


def compute_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute empirical coverage of a prediction interval.

    The interval is [y_pred - z * y_std, y_pred + z * y_std] where z is
    determined by the desired confidence level (normal approximation).

    Parameters
    ----------
    y_true:
        Ground-truth targets, shape [N].
    y_pred:
        Predicted means, shape [N].
    y_std:
        Predicted standard deviations, shape [N].
    confidence:
        Desired confidence level (e.g. 0.95 for 95% CI).

    Returns
    -------
    Fraction of true values falling within the interval.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    z = norm.ppf((1.0 + confidence) / 2.0)  # e.g. 1.96 for 95%
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std

    in_interval = (y_true >= lower) & (y_true <= upper)
    return float(in_interval.mean())


def compute_ood_auroc(
    in_domain_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray,
) -> float:
    """Compute AUROC for separating in-domain vs OOD samples by uncertainty.

    A good uncertainty estimator should assign higher uncertainty to OOD
    molecules.  This metric measures how well uncertainty scores separate
    the two groups (higher = better OOD detection).

    Parameters
    ----------
    in_domain_uncertainties:
        Uncertainty scores for in-domain molecules.
    ood_uncertainties:
        Uncertainty scores for out-of-domain molecules.

    Returns
    -------
    AUROC for OOD detection (label 0=in-domain, 1=OOD).
    """
    in_d = np.asarray(in_domain_uncertainties, dtype=np.float64)
    ood = np.asarray(ood_uncertainties, dtype=np.float64)

    scores = np.concatenate([in_d, ood])
    labels = np.concatenate([
        np.zeros(len(in_d), dtype=np.int32),
        np.ones(len(ood), dtype=np.int32),
    ])

    if len(np.unique(labels)) < 2:
        return 0.5

    return float(roc_auc_score(labels, scores))
