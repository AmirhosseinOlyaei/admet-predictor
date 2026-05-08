"""Atom-level attribution using Integrated Gradients (captum)."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch import Tensor

logger = logging.getLogger(__name__)


class _GraphWrapper(nn.Module):
    """Thin wrapper to expose node features as the IG input.

    captum's IntegratedGradients requires a single tensor input.
    This wrapper holds everything else fixed and varies node features.
    """

    def __init__(self, model: nn.Module, batch, task_name: str, task_configs: list[dict]) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.task_name = task_name
        self.task_type_map = {tc["name"]: tc["task_type"] for tc in task_configs}

    def forward(self, x: Tensor) -> Tensor:
        """Run the model with modified node features x."""
        self.batch.x = x
        preds = self.model(self.batch)
        pred = preds[self.task_name]
        task_type = self.task_type_map[self.task_name]
        if task_type == "classification":
            return torch.sigmoid(pred.squeeze(-1))
        else:
            # Return gamma (mean prediction)
            return pred[0].squeeze(-1)


class MolecularAttributor:
    """Compute atom-level attributions via Integrated Gradients.

    Parameters
    ----------
    model:
        Trained ADMETModel (or any nn.Module).
    task_configs:
        List of task config dicts.
    """

    def __init__(self, model: nn.Module, task_configs: list[dict]) -> None:
        self.model = model
        self.task_configs = task_configs

    def explain(self, smiles: str, task_name: str) -> dict[str, Any]:
        """Generate atom attributions for a single SMILES string.

        Parameters
        ----------
        smiles:
            Query SMILES string.
        task_name:
            Which ADMET task to explain.

        Returns
        -------
        Dict with keys:
        - atom_scores: list of floats, one per atom
        - smiles: input SMILES
        - image_base64: base64-encoded PNG of the molecule with atom coloring
        """
        from admet_predictor.data.featurize import mol_to_graph
        from torch_geometric.data import Batch, Data

        graph = mol_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        device = next(self.model.parameters()).device

        data = Data(
            x=graph["node_features"].to(device),
            edge_index=graph["edge_index"].to(device),
            edge_attr=graph["edge_features"].to(device),
        )
        data.smiles = smiles
        batch = Batch.from_data_list([data])
        # Re-attach smiles list (PyG batching drops string attributes)
        batch.smiles = [smiles]

        x_input = data.x.clone().requires_grad_(True)

        wrapper = _GraphWrapper(
            model=self.model,
            batch=batch,
            task_name=task_name,
            task_configs=self.task_configs,
        ).to(device)

        ig = IntegratedGradients(wrapper)
        baseline = torch.zeros_like(x_input)

        self.model.eval()
        attributions, _ = ig.attribute(
            x_input,
            baselines=baseline,
            n_steps=50,
            return_convergence_delta=True,
        )

        # Atom scores: sum across feature dimension
        atom_scores = attributions.sum(dim=-1).detach().cpu().numpy().tolist()

        # Draw molecule with atom coloring
        image_b64 = self._draw_molecule(smiles, atom_scores)

        return {
            "smiles": smiles,
            "task": task_name,
            "atom_scores": atom_scores,
            "image_base64": image_b64,
        }

    @staticmethod
    def _draw_molecule(smiles: str, atom_scores: list[float]) -> str:
        """Render molecule with per-atom colour intensity (base64 PNG)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        import numpy as np

        scores = np.array(atom_scores)
        abs_max = np.abs(scores).max()
        if abs_max > 0:
            norm_scores = scores / abs_max
        else:
            norm_scores = scores

        # Build colour dict: positive = green, negative = red
        atom_colors: dict[int, tuple] = {}
        highlight_atoms: list[int] = []

        for idx, score in enumerate(norm_scores):
            r = max(0.0, -score)
            g = max(0.0, score)
            b = 0.0
            atom_colors[idx] = (r, g, b)
            highlight_atoms.append(idx)

        drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
        drawer.drawOptions().addStereoAnnotation = False
        rdMolDraw2D.PrepareMolForDrawing(mol)

        try:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
                highlightBonds=[],
            )
        except Exception:
            # Fall back to plain drawing
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        svg_str = drawer.GetDrawingText()

        # Convert SVG to PNG via cairosvg if available; else return SVG as b64
        try:
            import cairosvg

            png_bytes = cairosvg.svg2png(bytestring=svg_str.encode())
            return base64.b64encode(png_bytes).decode()
        except ImportError:
            # Return SVG encoded as base64 (can be displayed in HTML with data: URI)
            return base64.b64encode(svg_str.encode()).decode()
