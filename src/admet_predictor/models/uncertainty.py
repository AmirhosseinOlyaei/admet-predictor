"""Uncertainty estimation utilities for ADMET predictions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
from torch import Tensor


@contextmanager
def enable_dropout(model: nn.Module) -> Generator[None, None, None]:
    """Context manager that enables Dropout layers while keeping the rest in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    try:
        yield
    finally:
        model.eval()


def mc_dropout_predict(
    model: nn.Module,
    batch,
    task_configs: list[dict],
    n_samples: int = 30,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Monte Carlo Dropout inference for uncertainty estimation.

    Parameters
    ----------
    model:
        The ADMETModel instance.
    batch:
        PyG Batch object.
    task_configs:
        List of task config dicts.
    n_samples:
        Number of stochastic forward passes.

    Returns
    -------
    Dict mapping task_name → (mean, variance/entropy) tensors.
    - Classification: mean is probability, second is predictive entropy.
    - Regression:     mean is gamma mean, second is sample variance.
    """
    task_names = [tc["name"] for tc in task_configs]
    task_types = {tc["name"]: tc["task_type"] for tc in task_configs}

    # Collect samples
    all_samples: dict[str, list[Tensor]] = {name: [] for name in task_names}

    with enable_dropout(model):
        with torch.no_grad():
            for _ in range(n_samples):
                preds = model(batch)
                for name, pred in preds.items():
                    if task_types[name] == "classification":
                        # pred is logit [B, 1]
                        prob = torch.sigmoid(pred).squeeze(-1)  # [B]
                        all_samples[name].append(prob)
                    else:
                        # pred is (gamma, nu, alpha, beta) each [B, 1]
                        gamma = pred[0].squeeze(-1)  # [B]
                        all_samples[name].append(gamma)

    results: dict[str, tuple[Tensor, Tensor]] = {}
    for name in task_names:
        stacked = torch.stack(all_samples[name], dim=0)  # [n_samples, B]
        mean = stacked.mean(dim=0)
        if task_types[name] == "classification":
            # Predictive entropy: H = -p*log(p) - (1-p)*log(1-p)
            p = mean.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            results[name] = (mean, entropy)
        else:
            variance = stacked.var(dim=0)
            results[name] = (mean, variance)

    return results


def nig_to_uncertainty(
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor]:
    """Decompose NIG output into aleatoric and epistemic uncertainty.

    Based on Amini et al. (2020) "Deep Evidential Regression".

    Parameters
    ----------
    gamma, nu, alpha, beta:
        NIG parameters from RegressionHead.

    Returns
    -------
    (aleatoric, epistemic) uncertainty tensors, same shape as gamma.
    """
    # Expected variance of the NIG distribution (data/aleatoric uncertainty)
    aleatoric = beta / (alpha - 1.0).clamp(min=1e-6)
    # Variance of the mean (model/epistemic uncertainty)
    epistemic = beta / (nu * (alpha - 1.0)).clamp(min=1e-6)
    return aleatoric, epistemic
