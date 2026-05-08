"""Per-task prediction heads for ADMET multi-task learning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RegressionHead(nn.Module):
    """Evidential regression head outputting Normal-Inverse-Gamma parameters.

    Outputs (gamma, nu, alpha, beta) where:
    - gamma: mean prediction (unconstrained)
    - nu:    precision weight  (softplus > 0)
    - alpha: shape parameter   (softplus + 1 > 1)
    - beta:  scale parameter   (softplus > 0)

    Parameters
    ----------
    hidden_dim:
        Input feature dimension.
    dropout:
        Dropout probability in the intermediate layer.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        self._softplus = nn.Softplus()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (gamma, nu, alpha, beta) each of shape [B, 1]."""
        out = self.net(x)  # [B, 4]
        gamma = out[:, 0:1]
        nu = self._softplus(out[:, 1:2])
        alpha = self._softplus(out[:, 2:3]) + 1.0
        beta = self._softplus(out[:, 3:4])
        return gamma, nu, alpha, beta


class ClassificationHead(nn.Module):
    """Binary classification head outputting a raw logit.

    Parameters
    ----------
    hidden_dim:
        Input feature dimension.
    dropout:
        Dropout probability in the intermediate layer.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return logit of shape [B, 1]."""
        return self.net(x)


class MultiTaskHeads(nn.Module):
    """Container for all per-task prediction heads.

    Parameters
    ----------
    task_configs:
        List of task config dicts with 'name' and 'task_type' keys.
    hidden_dim:
        Input dimension shared by all heads.
    dropout:
        Dropout probability for each head.
    """

    def __init__(
        self,
        task_configs: list[dict],
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.task_names = [tc["name"] for tc in task_configs]

        heads: dict[str, nn.Module] = {}
        for tc in task_configs:
            name = tc["name"]
            if tc["task_type"] == "regression":
                heads[name] = RegressionHead(hidden_dim=hidden_dim, dropout=dropout)
            else:
                heads[name] = ClassificationHead(hidden_dim=hidden_dim, dropout=dropout)
        self.heads = nn.ModuleDict(heads)

    def forward(
        self, x: Tensor
    ) -> dict[str, Tensor | tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Run all task heads and return predictions dict.

        Parameters
        ----------
        x:
            Shared trunk output, shape [B, hidden_dim].

        Returns
        -------
        Dict mapping task_name → tensor (classification) or 4-tuple (regression).
        """
        return {name: self.heads[name](x) for name in self.task_names}
