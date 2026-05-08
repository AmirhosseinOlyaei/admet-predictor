"""Unit tests for molecular featurization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from admet_predictor.data.featurize import (
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    mol_to_fingerprint,
    mol_to_graph,
)

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
# Aspirin: C9H8O4 → 13 heavy atoms, 13 bonds → 26 directed edges


class TestMolToGraph:
    def test_aspirin_node_count(self):
        result = mol_to_graph(ASPIRIN_SMILES)
        assert result is not None
        # Aspirin has 13 heavy atoms
        assert result["node_features"].shape == (13, ATOM_FEAT_DIM)

    def test_aspirin_edge_count(self):
        result = mol_to_graph(ASPIRIN_SMILES)
        assert result is not None
        # 13 bonds in aspirin (including the ring), each stored twice (directed)
        edge_index = result["edge_index"]
        assert edge_index.shape[0] == 2
        num_directed_edges = edge_index.shape[1]
        # Should be even (bidirectional)
        assert num_directed_edges % 2 == 0

    def test_node_feature_dim(self):
        result = mol_to_graph(ASPIRIN_SMILES)
        assert result is not None
        assert result["node_features"].shape[1] == ATOM_FEAT_DIM

    def test_edge_feature_dim(self):
        result = mol_to_graph(ASPIRIN_SMILES)
        assert result is not None
        n_edges = result["edge_index"].shape[1]
        assert result["edge_features"].shape == (n_edges, BOND_FEAT_DIM)

    def test_tensor_types(self):
        result = mol_to_graph(ASPIRIN_SMILES)
        assert result is not None
        assert result["node_features"].dtype == torch.float32
        assert result["edge_index"].dtype == torch.int64
        assert result["edge_features"].dtype == torch.float32

    def test_invalid_smiles_returns_none(self):
        assert mol_to_graph("NOT_A_SMILES") is None
        assert mol_to_graph("") is None
        assert mol_to_graph(None) is None  # type: ignore

    def test_single_atom(self):
        result = mol_to_graph("[He]")
        assert result is not None
        assert result["node_features"].shape[0] == 1
        assert result["edge_index"].shape[1] == 0

    def test_edge_index_valid(self):
        result = mol_to_graph(ASPIRIN_SMILES)
        assert result is not None
        n_atoms = result["node_features"].shape[0]
        assert result["edge_index"].max().item() < n_atoms
        assert result["edge_index"].min().item() >= 0


class TestMolToFingerprint:
    def test_correct_dimension(self):
        fp = mol_to_fingerprint(ASPIRIN_SMILES)
        assert fp is not None
        assert fp.shape == (2215,)

    def test_dtype(self):
        fp = mol_to_fingerprint(ASPIRIN_SMILES)
        assert fp is not None
        assert fp.dtype == np.float32

    def test_binary_values(self):
        fp = mol_to_fingerprint(ASPIRIN_SMILES)
        assert fp is not None
        unique_vals = set(np.unique(fp).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_invalid_smiles_returns_none(self):
        assert mol_to_fingerprint("INVALID") is None
        assert mol_to_fingerprint("") is None

    def test_different_molecules_differ(self):
        fp1 = mol_to_fingerprint(ASPIRIN_SMILES)
        fp2 = mol_to_fingerprint("Cn1cnc2c1c(=O)n(C)c(=O)n2C")  # caffeine
        assert fp1 is not None and fp2 is not None
        assert not np.array_equal(fp1, fp2)
