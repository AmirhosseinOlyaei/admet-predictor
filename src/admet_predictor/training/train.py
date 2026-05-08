"""Main training entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from admet_predictor.data.datamodule import ADMETDataModule
from admet_predictor.models.admet_model import ADMETModel
from admet_predictor.training.callbacks import CalibrationCallback, GradNormCallback

logger = logging.getLogger(__name__)


def train(
    config_path: str,
    data_config_path: str,
    data_dir: str,
    output_dir: str,
) -> None:
    """Train the ADMET model.

    Parameters
    ----------
    config_path:
        Path to model YAML config (e.g. configs/model/attentivefp_base.yaml).
    data_config_path:
        Path to data YAML config (e.g. configs/data/admet_tasks.yaml).
    data_dir:
        Directory containing processed parquet files.
    output_dir:
        Directory for checkpoints and logs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)
    task_configs = data_config["tasks"]

    logger.info("Loaded %d task configs", len(task_configs))

    # DataModule
    dm = ADMETDataModule(
        data_dir=data_dir,
        task_configs=task_configs,
        batch_size=128,
        num_workers=4,
    )
    dm.setup("fit")
    pos_weights = dm.compute_pos_weights()

    # Model
    model = ADMETModel(
        model_config=model_config,
        task_configs=task_configs,
        pos_weights=pos_weights,
    )

    # MLflow
    mlflow_uri = str(output_dir / "mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("admet_predictor")
    mlf_logger = MLFlowLogger(
        experiment_name="admet_predictor",
        tracking_uri=mlflow_uri,
        log_model=True,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="admet-{epoch:03d}-{val/composite_score:.4f}",
        monitor="val/composite_score",
        mode="max",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/composite_score",
        mode="max",
        patience=15,
        verbose=True,
    )
    gradnorm_cb = GradNormCallback(update_every=100)
    calibration_cb = CalibrationCallback(eval_every=5)

    # Trainer
    trainer = Trainer(
        max_epochs=100,
        logger=mlf_logger,
        callbacks=[checkpoint_cb, early_stop_cb, gradnorm_cb, calibration_cb],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision="16-mixed",
        deterministic=False,
    )

    logger.info("Starting training ...")
    trainer.fit(model, datamodule=dm)

    logger.info("Best checkpoint: %s", checkpoint_cb.best_model_path)
    logger.info("Best val/composite_score: %.4f", checkpoint_cb.best_model_score)
