"""GradNorm: Gradient Normalization for Multi-Task Learning (Chen et al., 2018)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GradNormLoss(nn.Module):
    """GradNorm adaptive task weighting.

    Reference:
        Chen, Z. et al. (2018). GradNorm: Gradient Normalization for
        Adaptive Loss Balancing in Deep Multitask Networks. ICML 2018.

    Parameters
    ----------
    num_tasks:
        Number of tasks.
    alpha:
        Restoring force hyperparameter (typically 0.12 ≤ α ≤ 3.0).
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        # Learnable per-task weights; initialise to 1
        self.task_weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))

    # ------------------------------------------------------------------
    # GradNorm update
    # ------------------------------------------------------------------

    def compute_gradnorm_loss(
        self,
        task_losses: list[Tensor],
        shared_params: list[Tensor],
        initial_losses: list[float],
    ) -> Tensor:
        """Compute the GradNorm loss for updating task weights.

        Parameters
        ----------
        task_losses:
            List of per-task scalar loss tensors (at current step).
        shared_params:
            Parameters of the last shared layer (used for gradient norms).
        initial_losses:
            Initial loss values for each task (used to compute relative progress).

        Returns
        -------
        GradNorm loss (scalar tensor).
        """
        # Weighted losses
        weighted = [self.task_weights[i] * task_losses[i] for i in range(self.num_tasks)]

        # Compute gradient norm for each task wrt shared params
        grad_norms: list[Tensor] = []
        for wl in weighted:
            grads = torch.autograd.grad(
                wl,
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            # L2 norm of all gradients concatenated
            valid_grads = [g.view(-1) for g in grads if g is not None]
            if valid_grads:
                norm = torch.cat(valid_grads).norm(p=2)
            else:
                norm = torch.tensor(0.0, device=self.task_weights.device)
            grad_norms.append(norm)

        grad_norms_t = torch.stack(grad_norms)  # [num_tasks]

        # Mean gradient norm across tasks (detached target)
        mean_norm = grad_norms_t.detach().mean()

        # Relative inverse training rate
        with torch.no_grad():
            current_losses = torch.stack(
                [tl.detach() for tl in task_losses]
            )
            init_t = torch.tensor(
                initial_losses, dtype=torch.float32, device=current_losses.device
            ).clamp(min=1e-8)
            loss_ratio = current_losses / init_t
            r_i = loss_ratio / loss_ratio.mean().clamp(min=1e-8)

        target_norms = (mean_norm * r_i**self.alpha).detach()
        gradnorm_loss = torch.abs(grad_norms_t - target_norms).sum()
        return gradnorm_loss

    # ------------------------------------------------------------------
    # Weighted combination
    # ------------------------------------------------------------------

    def get_weighted_loss(self, task_losses: list[Tensor]) -> Tensor:
        """Return weighted sum of task losses, with weights normalised to sum to num_tasks.

        Parameters
        ----------
        task_losses:
            List of per-task scalar loss tensors.

        Returns
        -------
        Scalar weighted total loss.
        """
        # Normalise weights so they sum to num_tasks (keeps scale stable)
        w = self.task_weights.clamp(min=1e-6)
        w_normalised = w / w.sum() * self.num_tasks

        total = sum(
            w_normalised[i] * task_losses[i] for i in range(len(task_losses))
        )
        return total
