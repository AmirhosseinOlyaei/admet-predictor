"""Applicability domain assessment using nearest-neighbour Tanimoto similarity."""

from __future__ import annotations

import logging

import faiss
import numpy as np

from admet_predictor.data.featurize import mol_to_fingerprint

logger = logging.getLogger(__name__)


def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto coefficient for binary bit-vector fingerprints."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    intersection = float(np.dot(a, b))
    union = float(np.sum(a) + np.sum(b) - intersection)
    return intersection / (union + 1e-10)


class ApplicabilityDomain:
    """k-NN applicability domain based on Morgan+MACCS fingerprints.

    A compound is considered within the applicability domain if its
    maximum Tanimoto similarity to any training molecule exceeds `threshold`.

    Parameters
    ----------
    k:
        Number of nearest neighbours to retrieve from the faiss index.
    threshold:
        Minimum Tanimoto similarity to be considered in-domain.
    """

    def __init__(self, k: int = 5, threshold: float = 0.3) -> None:
        self.k = k
        self.threshold = threshold
        self._index: faiss.Index | None = None
        self._train_fps: np.ndarray | None = None  # float32, [N, 2215]

    def fit(self, smiles_list: list[str]) -> None:
        """Build a faiss index from training SMILES fingerprints.

        Parameters
        ----------
        smiles_list:
            List of training SMILES strings.
        """
        fps: list[np.ndarray] = []
        for smi in smiles_list:
            fp = mol_to_fingerprint(smi)
            if fp is not None:
                fps.append(fp)

        if not fps:
            raise ValueError("No valid fingerprints computed from smiles_list")

        fp_matrix = np.stack(fps, axis=0).astype(np.float32)  # [N, 2215]
        self._train_fps = fp_matrix

        # faiss L2 index (we will post-hoc convert L2 to Tanimoto)
        dim = fp_matrix.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(fp_matrix)  # type: ignore[arg-type]
        logger.info("ApplicabilityDomain fitted with %d molecules", len(fps))

    def score(self, smiles: str) -> tuple[float, bool]:
        """Compute applicability domain score for a query molecule.

        Parameters
        ----------
        smiles:
            Query SMILES string.

        Returns
        -------
        (max_tanimoto, is_in_domain) tuple.
        """
        if self._index is None or self._train_fps is None:
            raise RuntimeError("Call fit() before score()")

        fp = mol_to_fingerprint(smiles)
        if fp is None:
            return 0.0, False

        query = fp.reshape(1, -1).astype(np.float32)
        k = min(self.k, self._index.ntotal)
        _, indices = self._index.search(query, k)  # type: ignore[arg-type]

        max_sim = 0.0
        for idx in indices[0]:
            if idx < 0:
                continue
            sim = _tanimoto(fp, self._train_fps[idx])
            if sim > max_sim:
                max_sim = sim

        return max_sim, max_sim >= self.threshold
