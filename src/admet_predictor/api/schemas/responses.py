"""Response schemas for the ADMET Predictor API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UncertaintyResult(BaseModel):
    aleatoric: Optional[float] = None
    epistemic: Optional[float] = None
    ci_95: Optional[tuple[float, float]] = None
    predictive_entropy: Optional[float] = None


class TaskPrediction(BaseModel):
    value: float
    unit: Optional[str] = None
    task_type: str  # "regression" or "classification"
    probability: Optional[float] = None
    predicted_class: Optional[str] = None
    uncertainty: Optional[UncertaintyResult] = None
    in_domain: Optional[bool] = None
    domain_score: Optional[float] = None


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    inchikey: Optional[str] = None
    valid: bool
    predictions: dict[str, TaskPrediction]
    admet_score: float
    model_version: str
    inference_ms: float


class BatchJobResponse(BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "done" | "failed"
    estimated_seconds: Optional[int] = None
    results: Optional[list[dict]] = None


class ExplainResponse(BaseModel):
    smiles: str
    task: str
    atom_scores: list[float]
    image_base64: str  # base64-encoded PNG (or SVG)


class ModelInfoResponse(BaseModel):
    model_version: str
    num_tasks: int
    task_names: list[str]
    hidden_dim: int
    chemberta_model: str
