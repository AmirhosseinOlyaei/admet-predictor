"""High-level inference API for ADMET prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch_geometric.data import Batch, Data

from admet_predictor.data.featurize import mol_to_graph
from admet_predictor.data.standardize import standardize_smiles
from admet_predictor.inference.applicability import ApplicabilityDomain
from admet_predictor.inference.attribution import MolecularAttributor
from admet_predictor.models.admet_model import ADMETModel
from admet_predictor.models.uncertainty import mc_dropout_predict, nig_to_uncertainty

logger = logging.getLogger(__name__)


class ADMETPredictor:
    """Production inference wrapper for the ADMET model.

    Parameters
    ----------
    model_path:
        Path to a PyTorch Lightning checkpoint (.ckpt).
    task_configs:
        List of task config dicts from admet_tasks.yaml.
    model_config:
        Model hyperparameter dict from attentivefp_base.yaml.
    device:
        'cuda', 'cpu', or 'auto'.
    training_smiles:
        Optional list of training SMILES used to fit applicability domain.
    """

    def __init__(
        self,
        model_path: str | Path,
        task_configs: list[dict],
        model_config: dict,
        device: str = "auto",
        training_smiles: Optional[list[str]] = None,
    ) -> None:
        self.task_configs = task_configs
        self.task_names = [tc["name"] for tc in task_configs]
        self.task_type_map = {tc["name"]: tc["task_type"] for tc in task_configs}
        self.task_unit_map = {tc["name"]: tc.get("unit") for tc in task_configs}

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        self.model = ADMETModel.load_from_checkpoint(
            str(model_path),
            model_config=model_config,
            task_configs=task_configs,
            map_location=self.device,
        )
        self.model.to(self.device)
        self.model.eval()

        # Applicability domain
        self.domain = ApplicabilityDomain()
        if training_smiles:
            self.domain.fit(training_smiles)

        # Attribution
        self.attributor = MolecularAttributor(
            model=self.model, task_configs=task_configs
        )

        # Model version from checkpoint
        self.model_version = Path(model_path).stem

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        smiles: str,
        tasks: Optional[list[str]] = None,
        return_uncertainty: bool = True,
        return_domain: bool = True,
    ) -> dict[str, Any]:
        """Predict ADMET properties for a single molecule.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        tasks:
            List of task names to predict, or None for all tasks.
        return_uncertainty:
            If True, include uncertainty estimates.
        return_domain:
            If True, include applicability domain score.

        Returns
        -------
        Prediction dict with per-task results.
        """
        canon = standardize_smiles(smiles)
        valid = canon is not None
        if not valid:
            return {
                "smiles": smiles,
                "canonical_smiles": smiles,
                "valid": False,
                "predictions": {},
                "admet_score": 0.0,
            }

        graph = mol_to_graph(canon)
        if graph is None:
            return {
                "smiles": smiles,
                "canonical_smiles": canon,
                "valid": False,
                "predictions": {},
                "admet_score": 0.0,
            }

        # Build PyG batch
        data = Data(
            x=graph["node_features"].to(self.device),
            edge_index=graph["edge_index"].to(self.device),
            edge_attr=graph["edge_features"].to(self.device),
        )
        data.smiles = canon
        batch = Batch.from_data_list([data])
        batch.smiles = [canon]

        # Forward pass (deterministic)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(batch)

        # MC Dropout for classification uncertainty
        mc_results: dict = {}
        if return_uncertainty:
            mc_results = mc_dropout_predict(
                self.model, batch, self.task_configs, n_samples=30
            )

        # Applicability domain
        domain_score: Optional[float] = None
        in_domain: Optional[bool] = None
        if return_domain and hasattr(self.domain, "_index") and self.domain._index is not None:
            domain_score, in_domain = self.domain.score(canon)

        # Assemble per-task results
        active_tasks = tasks if tasks and tasks != ["all"] else self.task_names
        task_predictions: dict[str, dict] = {}

        for name in active_tasks:
            if name not in self.task_type_map:
                continue

            pred = preds[name]
            task_type = self.task_type_map[name]
            unit = self.task_unit_map.get(name)
            result: dict[str, Any] = {
                "task_type": task_type,
                "unit": unit,
                "in_domain": in_domain,
                "domain_score": domain_score,
            }

            if task_type == "classification":
                logit = pred.squeeze().item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                result["value"] = prob
                result["probability"] = prob
                result["predicted_class"] = "positive" if prob >= 0.5 else "negative"

                if return_uncertainty and name in mc_results:
                    mc_mean, mc_entropy = mc_results[name]
                    result["uncertainty"] = {
                        "predictive_entropy": mc_entropy[0].item(),
                        "aleatoric": None,
                        "epistemic": None,
                        "ci_95": None,
                    }

            else:
                gamma, nu, alpha, beta = pred
                mean_val = gamma.squeeze().item()
                result["value"] = mean_val
                result["probability"] = None
                result["predicted_class"] = None

                if return_uncertainty:
                    ale, epi = nig_to_uncertainty(gamma, nu, alpha, beta)
                    ale_val = ale.squeeze().item()
                    epi_val = epi.squeeze().item()
                    total_std = (ale_val + epi_val) ** 0.5
                    result["uncertainty"] = {
                        "aleatoric": ale_val,
                        "epistemic": epi_val,
                        "ci_95": (mean_val - 1.96 * total_std, mean_val + 1.96 * total_std),
                        "predictive_entropy": None,
                    }

            if not return_uncertainty:
                result["uncertainty"] = None

            task_predictions[name] = result

        # InChIKey
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchiKey

            mol = Chem.MolFromSmiles(canon)
            inchikey = MolToInchiKey(mol) if mol else None
        except Exception:
            inchikey = None

        # Composite score (simple average of normalised scores)
        admet_score = self._compute_admet_score(task_predictions)

        return {
            "smiles": smiles,
            "canonical_smiles": canon,
            "inchikey": inchikey,
            "valid": True,
            "predictions": task_predictions,
            "admet_score": admet_score,
        }

    def predict_batch(
        self,
        smiles_list: list[str],
        batch_size: int = 64,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Predict ADMET properties for a list of molecules.

        Parameters
        ----------
        smiles_list:
            List of SMILES strings.
        batch_size:
            Number of molecules per inference batch.

        Returns
        -------
        List of prediction dicts, one per input SMILES.
        """
        results: list[dict] = []
        for i in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[i : i + batch_size]
            for smi in chunk:
                results.append(self.predict(smi, **kwargs))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_admet_score(self, task_predictions: dict[str, dict]) -> float:
        """Compute a simple composite ADMET score in [0, 1]."""
        scores: list[float] = []
        for name, result in task_predictions.items():
            task_type = result["task_type"]
            val = result.get("value", 0.0)
            if val is None:
                continue
            if task_type == "classification":
                scores.append(float(val))  # probability already in [0, 1]
            else:
                scores.append(1.0 / (1.0 + abs(float(val))))
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))
