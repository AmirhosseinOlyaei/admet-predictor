"""Graph neural network encoder using GATv2 with attentive readout."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_add_pool

from admet_predictor.data.featurize import ATOM_FEAT_DIM, BOND_FEAT_DIM


class GraphEncoder(nn.Module):
    """GATv2-based graph encoder with attentive global readout.

    Architecture
    ------------
    1. Atom embedding: Linear → LayerNorm → ReLU → Dropout
    2. N × GATv2Conv with residual connections and LayerNorm
    3. GlobalAttention readout → molecule representation [B, hidden_dim]

    Parameters
    ----------
    hidden_dim:
        Dimension of hidden representations.
    num_gat_layers:
        Number of GATv2 message-passing layers.
    num_attention_heads:
        Number of attention heads per GATv2 layer (concat=False).
    dropout_graph:
        Dropout probability used throughout the graph encoder.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_gat_layers: int = 5,
        num_attention_heads: int = 8,
        dropout_graph: float = 0.15,
        atom_feat_dim: int = ATOM_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers

        # Atom embedding
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_graph),
        )

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for _ in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_attention_heads,
                    concat=False,
                    dropout=dropout_graph,
                    edge_dim=BOND_FEAT_DIM,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            # Residual projection (identity since dims match)
            self.residual_projs.append(nn.Identity())

        self.dropout = nn.Dropout(dropout_graph)

        # GlobalAttention readout: gate network + node feats → pooled graph repr
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.readout = GlobalAttention(gate_nn=gate_nn, nn=None)

    def forward(self, batch) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        batch:
            PyG Batch object with .x, .edge_index, .edge_attr, .batch.

        Returns
        -------
        Tensor of shape [batch_size, hidden_dim].
        """
        x: Tensor = batch.x
        edge_index: Tensor = batch.edge_index
        edge_attr: Tensor = batch.edge_attr
        batch_vec: Tensor = batch.batch

        # Embed atoms
        x = self.atom_embedding(x)

        # GATv2 layers with residual connections
        for gat, norm, proj in zip(
            self.gat_layers, self.layer_norms, self.residual_projs
        ):
            residual = proj(x)
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = norm(x + residual)

        # Global attentive readout → [batch_size, hidden_dim]
        graph_repr = self.readout(x, batch_vec)
        return graph_repr
