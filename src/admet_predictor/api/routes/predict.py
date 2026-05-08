"""Single-molecule prediction endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from admet_predictor.api.schemas.requests import PredictRequest
from admet_predictor.api.schemas.responses import PredictResponse, TaskPrediction, UncertaintyResult

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Predict ADMET properties for a single molecule.

    The SMILES is validated and canonicalized by Pydantic. If the predictor
    is not loaded (e.g. in test mode), a 503 is returned.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    result = predictor.predict(
        smiles=body.smiles,
        tasks=body.tasks,
        return_uncertainty=body.return_uncertainty,
        return_domain=body.applicability_domain,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not result.get("valid", False):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid or unparseable SMILES: {body.smiles!r}",
        )

    # Convert raw prediction dicts to TaskPrediction models
    task_preds: dict[str, TaskPrediction] = {}
    for task_name, raw in result.get("predictions", {}).items():
        unc_raw = raw.get("uncertainty")
        uncertainty = None
        if unc_raw is not None:
            uncertainty = UncertaintyResult(
                aleatoric=unc_raw.get("aleatoric"),
                epistemic=unc_raw.get("epistemic"),
                ci_95=tuple(unc_raw["ci_95"]) if unc_raw.get("ci_95") else None,
                predictive_entropy=unc_raw.get("predictive_entropy"),
            )
        task_preds[task_name] = TaskPrediction(
            value=float(raw["value"]),
            unit=raw.get("unit"),
            task_type=raw["task_type"],
            probability=raw.get("probability"),
            predicted_class=raw.get("predicted_class"),
            uncertainty=uncertainty,
            in_domain=raw.get("in_domain"),
            domain_score=raw.get("domain_score"),
        )

    return PredictResponse(
        smiles=body.smiles,
        canonical_smiles=result["canonical_smiles"],
        inchikey=result.get("inchikey"),
        valid=True,
        predictions=task_preds,
        admet_score=float(result.get("admet_score", 0.0)),
        model_version=getattr(predictor, "model_version", "unknown"),
        inference_ms=elapsed_ms,
    )
