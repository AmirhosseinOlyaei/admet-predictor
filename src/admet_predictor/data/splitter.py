"""Scaffold-based train/val/test splitter."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


class ScaffoldSplitter:
    """Split a dataset by Murcko scaffold for a more challenging evaluation.

    Molecules sharing a scaffold are kept together in the same split, which
    tests generalisation to structurally novel compounds.
    """

    def split(
        self,
        smiles_list: list[str],
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return (train_indices, val_indices, test_indices).

        Parameters
        ----------
        smiles_list:
            List of SMILES strings.
        frac_train, frac_val, frac_test:
            Fraction sizes; must sum to 1.0.
        seed:
            Random seed for deterministic splits when scaffolds tie.
        """
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
            "Fractions must sum to 1"
        )

        rng = np.random.default_rng(seed)

        # Compute Murcko scaffold for each molecule
        scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            scaffold = self._murcko_scaffold(smi)
            scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds: largest first (most challenging split)
        scaffold_sets = sorted(
            scaffold_to_indices.values(), key=lambda x: len(x), reverse=True
        )

        n = len(smiles_list)
        n_train = int(np.floor(frac_train * n))
        n_val = int(np.floor(frac_val * n))

        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []

        for scaffold_set in scaffold_sets:
            if len(train_idx) + len(scaffold_set) <= n_train:
                train_idx.extend(scaffold_set)
            elif len(val_idx) + len(scaffold_set) <= n_val:
                val_idx.extend(scaffold_set)
            else:
                test_idx.extend(scaffold_set)

        logger.info(
            "Scaffold split: train=%d, val=%d, test=%d (total=%d)",
            len(train_idx),
            len(val_idx),
            len(test_idx),
            n,
        )
        return train_idx, val_idx, test_idx

    @staticmethod
    def _murcko_scaffold(smiles: str) -> str:
        """Compute the Murcko scaffold SMILES for a molecule.

        Returns an empty string for invalid SMILES (all grouped together).
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            return scaffold
        except Exception:
            return ""
