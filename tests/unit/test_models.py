"""Unit tests for model components."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from admet_predictor.data.featurize import ATOM_FEAT_DIM, BOND_FEAT_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pyg_batch(smiles_list: list[str]) -> Batch:
    """Build a minimal PyG Batch from SMILES."""
    from admet_predictor.data.featurize import mol_to_graph

    data_list = []
    for smi in smiles_list:
        g = mol_to_graph(smi)
        assert g is not None
        d = Data(
            x=g["node_features"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_features"],
        )
        d.smiles = smi
        data_list.append(d)

    batch = Batch.from_data_list(data_list)
    batch.smiles = smiles_list
    return batch


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
SMILES_LIST = [ASPIRIN, CAFFEINE]

HIDDEN_DIM = 64  # small for speed


class TestGraphEncoder:
    def test_output_shape(self):
        from admet_predictor.models.graph_encoder import GraphEncoder

        encoder = GraphEncoder(
            hidden_dim=HIDDEN_DIM,
            num_gat_layers=2,
            num_attention_heads=4,
            dropout_graph=0.0,
        )
        batch = _make_pyg_batch(SMILES_LIST)
        encoder.eval()
        with torch.no_grad():
            out = encoder(batch)
        assert out.shape == (len(SMILES_LIST), HIDDEN_DIM)

    def test_single_molecule(self):
        from admet_predictor.models.graph_encoder import GraphEncoder

        encoder = GraphEncoder(
            hidden_dim=HIDDEN_DIM,
            num_gat_layers=2,
            num_attention_heads=4,
            dropout_graph=0.0,
        )
        batch = _make_pyg_batch([ASPIRIN])
        encoder.eval()
        with torch.no_grad():
            out = encoder(batch)
        assert out.shape == (1, HIDDEN_DIM)

    def test_no_nan_output(self):
        from admet_predictor.models.graph_encoder import GraphEncoder

        encoder = GraphEncoder(hidden_dim=HIDDEN_DIM, num_gat_layers=2, num_attention_heads=4)
        batch = _make_pyg_batch(SMILES_LIST)
        encoder.eval()
        with torch.no_grad():
            out = encoder(batch)
        assert not torch.isnan(out).any()


class TestGatedFusion:
    def test_output_shape(self):
        from admet_predictor.models.fusion import GatedFusion

        fusion = GatedFusion(hidden_dim=HIDDEN_DIM)
        B = 4
        graph_repr = torch.randn(B, HIDDEN_DIM)
        bert_repr = torch.randn(B, HIDDEN_DIM)
        out = fusion(graph_repr, bert_repr)
        assert out.shape == (B, HIDDEN_DIM)

    def test_gate_bounds(self):
        """Output should be a convex combination → within [min, max] of inputs."""
        from admet_predictor.models.fusion import GatedFusion

        fusion = GatedFusion(hidden_dim=HIDDEN_DIM)
        fusion.eval()
        B = 8
        graph_repr = torch.ones(B, HIDDEN_DIM)
        bert_repr = torch.zeros(B, HIDDEN_DIM)
        with torch.no_grad():
            out = fusion(graph_repr, bert_repr)
        # Output should be in [0, 1] since it's a gate blend of 0 and 1
        assert (out >= -1e-5).all() and (out <= 1 + 1e-5).all()


class TestRegressionHead:
    def test_nig_constraints(self):
        """nu > 0, alpha > 1, beta > 0 must always hold."""
        from admet_predictor.models.task_heads import RegressionHead

        head = RegressionHead(hidden_dim=HIDDEN_DIM, dropout=0.0)
        head.eval()
        x = torch.randn(16, HIDDEN_DIM)
        with torch.no_grad():
            gamma, nu, alpha, beta = head(x)
        assert (nu > 0).all(), "nu must be > 0"
        assert (alpha > 1).all(), "alpha must be > 1"
        assert (beta > 0).all(), "beta must be > 0"

    def test_output_shapes(self):
        from admet_predictor.models.task_heads import RegressionHead

        head = RegressionHead(hidden_dim=HIDDEN_DIM)
        x = torch.randn(8, HIDDEN_DIM)
        gamma, nu, alpha, beta = head(x)
        for t in (gamma, nu, alpha, beta):
            assert t.shape == (8, 1)


class TestClassificationHead:
    def test_output_shape(self):
        from admet_predictor.models.task_heads import ClassificationHead

        head = ClassificationHead(hidden_dim=HIDDEN_DIM)
        x = torch.randn(8, HIDDEN_DIM)
        logits = head(x)
        assert logits.shape == (8, 1)

    def test_unconstrained_logits(self):
        """Logits can be any real number (sigmoid applied downstream)."""
        from admet_predictor.models.task_heads import ClassificationHead

        head = ClassificationHead(hidden_dim=HIDDEN_DIM)
        x = torch.randn(32, HIDDEN_DIM)
        logits = head(x)
        # Probabilities via sigmoid are always in (0, 1)
        probs = torch.sigmoid(logits)
        assert (probs > 0).all() and (probs < 1).all()
