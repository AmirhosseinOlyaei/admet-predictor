"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Molecule fixtures
# ---------------------------------------------------------------------------

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture
def sample_smiles() -> list[str]:
    """Return a list of well-known drug-like SMILES."""
    return [ASPIRIN_SMILES, CAFFEINE_SMILES, IBUPROFEN_SMILES]


@pytest.fixture
def aspirin_smiles() -> str:
    return ASPIRIN_SMILES


# ---------------------------------------------------------------------------
# Task configs fixture (subset for fast tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def task_configs() -> list[dict]:
    return [
        {
            "name": "herg",
            "tdc_name": "hERG",
            "task_type": "classification",
            "metric": "AUROC",
            "tdc_group": "Tox",
            "weight": 2.0,
            "unit": None,
        },
        {
            "name": "caco2_wang",
            "tdc_name": "Caco2_Wang",
            "task_type": "regression",
            "metric": "MAE",
            "tdc_group": "ADME",
            "weight": 1.0,
            "unit": "cm/s",
        },
    ]


# ---------------------------------------------------------------------------
# Mock predictor
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_predictor(task_configs):
    """Return a mock ADMETPredictor that returns synthetic predictions."""
    predictor = MagicMock()
    predictor.model_version = "test-v0.1"
    predictor.task_configs = task_configs
    predictor.task_names = [tc["name"] for tc in task_configs]

    def _predict(smiles, tasks=None, return_uncertainty=True, return_domain=True, **kw):
        from admet_predictor.data.standardize import standardize_smiles

        canon = standardize_smiles(smiles)
        if canon is None:
            return {"smiles": smiles, "canonical_smiles": smiles, "valid": False, "predictions": {}, "admet_score": 0.0}
        return {
            "smiles": smiles,
            "canonical_smiles": canon,
            "inchikey": None,
            "valid": True,
            "predictions": {
                "herg": {
                    "value": 0.3,
                    "unit": None,
                    "task_type": "classification",
                    "probability": 0.3,
                    "predicted_class": "negative",
                    "uncertainty": {
                        "aleatoric": None,
                        "epistemic": None,
                        "ci_95": None,
                        "predictive_entropy": 0.88,
                    } if return_uncertainty else None,
                    "in_domain": True,
                    "domain_score": 0.75,
                },
                "caco2_wang": {
                    "value": -5.2,
                    "unit": "cm/s",
                    "task_type": "regression",
                    "probability": None,
                    "predicted_class": None,
                    "uncertainty": {
                        "aleatoric": 0.12,
                        "epistemic": 0.05,
                        "ci_95": (-5.7, -4.7),
                        "predictive_entropy": None,
                    } if return_uncertainty else None,
                    "in_domain": True,
                    "domain_score": 0.75,
                },
            },
            "admet_score": 0.65,
        }

    predictor.predict.side_effect = _predict
    return predictor


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(mock_predictor):
    """Return a FastAPI TestClient with a mock predictor injected."""
    from admet_predictor.api.main import app

    # Patch the lifespan so it doesn't try to load a real checkpoint
    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.predictor = mock_predictor
        yield client
