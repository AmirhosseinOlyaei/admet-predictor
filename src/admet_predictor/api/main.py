"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from admet_predictor.api.middleware import RateLimitMiddleware, TimingMiddleware
from admet_predictor.api.routes import batch, explain, predict
from admet_predictor.api.schemas.responses import ModelInfoResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration paths (override with environment variables in production)
# ---------------------------------------------------------------------------
import os

_MODEL_CHECKPOINT = os.environ.get("ADMET_CHECKPOINT", "")
_DATA_CONFIG = os.environ.get(
    "ADMET_DATA_CONFIG",
    str(Path(__file__).parent.parent.parent.parent.parent / "configs/data/admet_tasks.yaml"),
)
_MODEL_CONFIG = os.environ.get(
    "ADMET_MODEL_CONFIG",
    str(
        Path(__file__).parent.parent.parent.parent.parent
        / "configs/model/attentivefp_base.yaml"
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ADMET predictor at startup; clean up at shutdown."""
    app.state.predictor = None

    if _MODEL_CHECKPOINT and Path(_MODEL_CHECKPOINT).exists():
        try:
            with open(_DATA_CONFIG) as f:
                data_cfg = yaml.safe_load(f)
            with open(_MODEL_CONFIG) as f:
                model_cfg = yaml.safe_load(f)

            from admet_predictor.inference.predictor import ADMETPredictor

            app.state.predictor = ADMETPredictor(
                model_path=_MODEL_CHECKPOINT,
                task_configs=data_cfg["tasks"],
                model_config=model_cfg,
            )
            logger.info("ADMETPredictor loaded from %s", _MODEL_CHECKPOINT)
        except Exception as exc:
            logger.error("Failed to load predictor: %s", exc)
    else:
        logger.warning(
            "No model checkpoint configured (ADMET_CHECKPOINT env var). "
            "Prediction endpoints will return 503."
        )

    yield

    app.state.predictor = None
    logger.info("ADMET Predictor shut down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ADMET Predictor API",
    description=(
        "Multi-task ADMET (Absorption, Distribution, Metabolism, "
        "Excretion, Toxicity) prediction for drug discovery."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware (order matters: outer → inner)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Routers
PREFIX = "/api/v1/admet"
app.include_router(predict.router, prefix=PREFIX, tags=["prediction"])
app.include_router(batch.router, prefix=PREFIX, tags=["batch"])
app.include_router(explain.router, prefix=PREFIX, tags=["explanation"])


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Health check endpoint."""
    predictor_ok = app.state.predictor is not None
    return {
        "status": "ok" if predictor_ok else "degraded",
        "predictor_loaded": predictor_ok,
    }


@app.get(f"{PREFIX}/model/info", response_model=ModelInfoResponse, tags=["metadata"])
async def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded model."""
    predictor = app.state.predictor

    if predictor is None:
        # Return placeholder info when model is not loaded
        return ModelInfoResponse(
            model_version="not_loaded",
            num_tasks=22,
            task_names=[],
            hidden_dim=256,
            chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
        )

    model_cfg = predictor.model.hparams.get("model_config", {})
    return ModelInfoResponse(
        model_version=predictor.model_version,
        num_tasks=len(predictor.task_configs),
        task_names=predictor.task_names,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        chemberta_model=model_cfg.get(
            "chemberta_model", "seyonec/ChemBERTa-zinc-base-v1"
        ),
    )
