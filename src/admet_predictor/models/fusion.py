"""Gated fusion of graph and BERT molecular representations."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GatedFusion(nn.Module):
    """Soft-gate fusion of two molecular representations.

    gate = sigmoid(W · [graph_repr || bert_repr])
    fused = gate * graph_repr + (1 - gate) * bert_repr

    Parameters
    ----------
    hidden_dim:
        Dimensionality of both input representations.
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, graph_repr: Tensor, bert_repr: Tensor) -> Tensor:
        """Fuse two representations.

        Parameters
        ----------
        graph_repr:
            Graph encoder output, shape [B, hidden_dim].
        bert_repr:
            BERT encoder output, shape [B, hidden_dim].

        Returns
        -------
        Fused tensor of shape [B, hidden_dim].
        """
        combined = torch.cat([graph_repr, bert_repr], dim=-1)  # [B, 2D]
        gate = torch.sigmoid(self.gate_linear(combined))        # [B, D]
        return gate * graph_repr + (1.0 - gate) * bert_repr
