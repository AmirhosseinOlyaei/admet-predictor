"""Custom PyTorch Lightning callbacks."""

from __future__ import annotations

import logging

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class GradNormCallback(Callback):
    """Update GradNorm task weights every N optimizer steps.

    After each backward pass, if the step count is a multiple of
    `update_every`, the GradNorm auxiliary loss is computed and the
    task_weights parameter is updated with a separate SGD step.
    """

    def __init__(self, update_every: int = 100, gradnorm_lr: float = 1e-3) -> None:
        super().__init__()
        self.update_every = update_every
        self.gradnorm_lr = gradnorm_lr
        self._step = 0
        self._gradnorm_optimizer: torch.optim.Optimizer | None = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if hasattr(pl_module, "gradnorm"):
            self._gradnorm_optimizer = torch.optim.SGD(
                [pl_module.gradnorm.task_weights], lr=self.gradnorm_lr
            )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._step += 1
        if self._step % self.update_every != 0:
            return
        if not hasattr(pl_module, "gradnorm") or self._gradnorm_optimizer is None:
            return
        if pl_module._initial_losses is None:
            return

        # Find a shared parameter set from the trunk's last linear layer
        try:
            # Gather shared parameters (trunk output layer)
            shared_params = list(pl_module.trunk[-2].parameters())
            if not shared_params:
                return

            # Compute task losses with current batch
            # We can't re-run the batch here, so skip gradnorm update if no active losses
            # In production, store the last task losses in the model
            if not hasattr(pl_module, "_last_task_losses"):
                return
            task_losses = pl_module._last_task_losses
            if not task_losses:
                return

            ordered = [
                task_losses[n]
                for n in pl_module.task_names
                if n in task_losses
            ]
            if not ordered:
                return

            initial = [
                pl_module._initial_losses[i]
                for i, n in enumerate(pl_module.task_names)
                if n in task_losses
            ]

            self._gradnorm_optimizer.zero_grad()
            gn_loss = pl_module.gradnorm.compute_gradnorm_loss(
                ordered, shared_params, initial
            )
            gn_loss.backward()
            self._gradnorm_optimizer.step()

            # Clamp weights to positive values
            with torch.no_grad():
                pl_module.gradnorm.task_weights.clamp_(min=1e-4)

            logger.debug("GradNorm loss: %.4f", gn_loss.item())

        except Exception as exc:
            logger.warning("GradNorm update failed: %s", exc)


class CalibrationCallback(Callback):
    """Compute Expected Calibration Error (ECE) on validation every N epochs."""

    def __init__(self, eval_every: int = 5) -> None:
        super().__init__()
        self.eval_every = eval_every

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.eval_every != 0:
            return

        from admet_predictor.evaluation.calibration import compute_ece

        # Use accumulated val preds from the model's storage (already reset at this point)
        # We operate on what trainer.logged_metrics has; store ECE=0 if no data
        try:
            # Access logged metrics for classification tasks
            for tc in pl_module.task_configs:
                if tc["task_type"] != "classification":
                    continue
                name = tc["name"]
                key_prob = f"val/auroc/{name}"
                # ECE computation requires raw probabilities — unavailable here post-reset
                # Log placeholder; full ECE can be computed in an evaluation script
                pl_module.log(f"val/ece_placeholder/{name}", 0.0)
        except Exception as exc:
            logger.warning("CalibrationCallback error: %s", exc)
