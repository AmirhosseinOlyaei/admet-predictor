"""Batch prediction endpoints using Celery for async processing."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from admet_predictor.api.schemas.requests import BatchPredictRequest
from admet_predictor.api.schemas.responses import BatchJobResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory job store (for development; replace with Redis in production)
# ---------------------------------------------------------------------------
_job_store: dict[str, dict[str, Any]] = {}


def _get_celery_app():
    """Lazily import celery app to avoid circular imports."""
    try:
        from celery import Celery  # type: ignore

        app = Celery(
            "admet_tasks",
            broker="redis://localhost:6379/0",
            backend="redis://localhost:6379/1",
        )
        return app
    except Exception:
        return None


@router.post("/predict/batch", response_model=BatchJobResponse)
async def submit_batch(request: Request, body: BatchPredictRequest) -> BatchJobResponse:
    """Submit a batch prediction job.

    Returns a job_id that can be polled via GET /predict/batch/{job_id}.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    job_id = str(uuid.uuid4())
    estimated_seconds = max(1, len(body.smiles_list) // 100)

    # Try to use Celery; fall back to synchronous processing stored in memory
    celery_app = _get_celery_app()
    if celery_app is not None:
        # Enqueue async task
        _job_store[job_id] = {"status": "queued", "results": None}
        # In a real deployment, this would call: batch_predict_task.delay(job_id, ...)
        # For now, store the job as queued
    else:
        # Synchronous fallback: run immediately
        results = predictor.predict_batch(
            smiles_list=body.smiles_list,
            tasks=body.tasks,
            return_uncertainty=body.return_uncertainty,
        )
        _job_store[job_id] = {"status": "done", "results": results}
        estimated_seconds = None

    return BatchJobResponse(
        job_id=job_id,
        status=_job_store[job_id]["status"],
        estimated_seconds=estimated_seconds,
    )


@router.get("/predict/batch/{job_id}", response_model=BatchJobResponse)
async def get_batch_result(job_id: str) -> BatchJobResponse:
    """Poll a batch prediction job by job_id.

    Returns status and, when done, the list of prediction results.
    """
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    job = _job_store[job_id]
    return BatchJobResponse(
        job_id=job_id,
        status=job["status"],
        results=job.get("results"),
    )
