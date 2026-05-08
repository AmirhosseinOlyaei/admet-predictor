"""Atom attribution / explanation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from admet_predictor.api.schemas.requests import ExplainRequest
from admet_predictor.api.schemas.responses import ExplainResponse

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: Request, body: ExplainRequest) -> ExplainResponse:
    """Generate atom-level attribution scores for a molecule/task pair.

    Uses Integrated Gradients (captum) to attribute the prediction to
    individual atoms.  Returns a base64-encoded PNG (or SVG) of the
    molecule coloured by attribution score.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.attributor.explain(
            smiles=body.smiles,
            task_name=body.task,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Attribution failed: {exc}",
        ) from exc

    return ExplainResponse(
        smiles=body.smiles,
        task=body.task,
        atom_scores=result["atom_scores"],
        image_base64=result["image_base64"],
    )
