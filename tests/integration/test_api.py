"""Integration tests for the ADMET Predictor API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


class TestHealth:
    def test_health_ok_with_predictor(self, test_client: TestClient):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["predictor_loaded"] is True

    def test_health_degraded_without_predictor(self):
        from admet_predictor.api.main import app

        with TestClient(app) as client:
            app.state.predictor = None
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["predictor_loaded"] is False


class TestPredictEndpoint:
    def test_predict_aspirin(self, test_client: TestClient):
        resp = test_client.post(
            "/api/v1/admet/predict",
            json={
                "smiles": ASPIRIN_SMILES,
                "tasks": ["all"],
                "return_uncertainty": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["valid"] is True
        assert "canonical_smiles" in data
        assert "predictions" in data
        assert "admet_score" in data
        assert isinstance(data["admet_score"], float)
        assert "inference_ms" in data

    def test_predict_returns_task_predictions(self, test_client: TestClient):
        resp = test_client.post(
            "/api/v1/admet/predict",
            json={"smiles": ASPIRIN_SMILES},
        )
        assert resp.status_code == 200
        data = resp.json()
        predictions = data["predictions"]
        assert len(predictions) > 0

        # Check structure of one prediction
        for task_name, pred in predictions.items():
            assert "value" in pred
            assert "task_type" in pred
            assert pred["task_type"] in ("classification", "regression")
            break

    def test_predict_invalid_smiles_returns_422(self, test_client: TestClient):
        resp = test_client.post(
            "/api/v1/admet/predict",
            json={"smiles": "THIS_IS_NOT_A_SMILES"},
        )
        assert resp.status_code == 422

    def test_predict_empty_smiles_returns_422(self, test_client: TestClient):
        resp = test_client.post(
            "/api/v1/admet/predict",
            json={"smiles": ""},
        )
        assert resp.status_code == 422

    def test_predict_without_predictor_returns_503(self):
        from admet_predictor.api.main import app

        with TestClient(app) as client:
            app.state.predictor = None
            resp = client.post(
                "/api/v1/admet/predict",
                json={"smiles": ASPIRIN_SMILES},
            )
            assert resp.status_code == 503

    def test_predict_model_version_in_response(self, test_client: TestClient):
        resp = test_client.post(
            "/api/v1/admet/predict",
            json={"smiles": ASPIRIN_SMILES},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)


class TestModelInfo:
    def test_model_info_endpoint(self, test_client: TestClient):
        resp = test_client.get("/api/v1/admet/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data
        assert "num_tasks" in data
        assert "task_names" in data
        assert isinstance(data["task_names"], list)
