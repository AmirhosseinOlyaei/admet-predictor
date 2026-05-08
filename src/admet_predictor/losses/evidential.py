"""Deep Evidential Regression loss (Amini et al., 2020)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def evidential_regression_loss(
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    y: Tensor,
    coeff: float = 1e-3,
) -> Tensor:
    """Compute the NIG evidential regression loss.

    Based on:
        Amini, A. et al. (2020). Deep Evidential Regression.
        NeurIPS 2020. https://arxiv.org/abs/1910.02600

    Parameters
    ----------
    gamma:
        Predicted mean, shape [B, 1] or [B].
    nu:
        Evidence for mean precision (> 0), same shape as gamma.
    alpha:
        Shape parameter (> 1), same shape as gamma.
    beta:
        Scale parameter (> 0), same shape as gamma.
    y:
        Ground-truth targets, shape [B] or [B, 1].
    coeff:
        Regularisation coefficient lambda.

    Returns
    -------
    Scalar loss tensor.
    """
    # Flatten to 1-D
    gamma = gamma.view(-1)
    nu = nu.view(-1)
    alpha = alpha.view(-1)
    beta = beta.view(-1)
    y = y.view(-1)

    two_b_lambda = 2.0 * beta * (1.0 + nu)

    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=gamma.device) / nu)
        - alpha * torch.log(two_b_lambda)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_b_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    reg = torch.abs(y - gamma) * (2.0 * nu + alpha)

    loss = nll + coeff * reg
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    return loss.mean()
