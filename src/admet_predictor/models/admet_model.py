"""Main ADMET multi-task Lightning model."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import MeanAbsoluteError, SpearmanCorrCoef

from admet_predictor.losses.evidential import evidential_regression_loss
from admet_predictor.losses.gradnorm import GradNormLoss
from admet_predictor.models.bert_encoder import BertEncoder
from admet_predictor.models.fusion import GatedFusion
from admet_predictor.models.graph_encoder import GraphEncoder
from admet_predictor.models.task_heads import MultiTaskHeads

logger = logging.getLogger(__name__)


class ADMETModel(LightningModule):
    """Multi-task ADMET prediction model.

    Architecture
    ------------
    GraphEncoder → \\
                    GatedFusion → SharedTrunk → MultiTaskHeads
    BertEncoder  → /

    Parameters
    ----------
    model_config:
        Dict from attentivefp_base.yaml.
    task_configs:
        List of task config dicts from admet_tasks.yaml.
    learning_rates:
        Dict with keys 'graph', 'bert', 'rest'. Defaults apply if missing.
    pos_weights:
        Dict mapping classification task_name → pos_weight tensor.
    """

    def __init__(
        self,
        model_config: dict,
        task_configs: list[dict],
        learning_rates: Optional[dict] = None,
        pos_weights: Optional[dict[str, Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weights"])

        self.task_configs = task_configs
        self.task_names = [tc["name"] for tc in task_configs]
        self.task_type_map = {tc["name"]: tc["task_type"] for tc in task_configs}

        cfg = model_config
        hidden_dim: int = cfg.get("hidden_dim", 256)
        self.hidden_dim = hidden_dim
        self.freeze_epochs: int = cfg.get("chemberta_freeze_epochs", 5)

        lr = learning_rates or {}
        self.lr_graph = lr.get("graph", 3e-4)
        self.lr_bert = lr.get("bert", 3e-5)
        self.lr_rest = lr.get("rest", 1e-3)

        # Encoders
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            num_gat_layers=cfg.get("num_gat_layers", 5),
            num_attention_heads=cfg.get("num_attention_heads", 8),
            dropout_graph=cfg.get("dropout_graph", 0.15),
        )
        self.bert_encoder = BertEncoder(
            model_name=cfg.get("chemberta_model", "seyonec/ChemBERTa-zinc-base-v1"),
            hidden_dim=hidden_dim,
        )

        # Fusion
        self.fusion = GatedFusion(hidden_dim=hidden_dim)

        # Shared trunk
        trunk_drop = cfg.get("dropout_trunk", 0.20)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(trunk_drop),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.Dropout(trunk_drop / 2),
        )

        # Task heads
        self.task_heads = MultiTaskHeads(
            task_configs=task_configs,
            hidden_dim=hidden_dim,
            dropout=cfg.get("dropout_heads", 0.10),
        )

        # GradNorm
        self.gradnorm = GradNormLoss(num_tasks=len(task_configs))

        # Positive class weights for BCE
        if pos_weights:
            self._pos_weights = {k: v for k, v in pos_weights.items()}
        else:
            self._pos_weights = {}

        # Metrics for validation
        self._val_preds: dict[str, list] = {n: [] for n in self.task_names}
        self._val_targets: dict[str, list] = {n: [] for n in self.task_names}

        # Initial losses for GradNorm (set after first batch)
        self._initial_losses: Optional[list[float]] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch) -> dict[str, Any]:
        """Run encoder→fusion→trunk→heads.

        Returns dict mapping task_name → prediction.
        Classification: Tensor [B, 1] (logit)
        Regression:     tuple (gamma, nu, alpha, beta) each [B, 1]
        """
        graph_repr = self.graph_encoder(batch)
        smiles_list: list[str] = batch.smiles
        bert_repr = self.bert_encoder(smiles_list)
        fused = self.fusion(graph_repr, bert_repr)
        trunk_out = self.trunk(fused)
        return self.task_heads(trunk_out)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_task_losses(
        self,
        preds: dict[str, Any],
        batch,
    ) -> dict[str, Tensor]:
        y_all: Tensor = batch.y          # [B, num_tasks]
        task_losses: dict[str, Tensor] = {}

        for task_idx, tc in enumerate(self.task_configs):
            name = tc["name"]
            pred = preds[name]

            y_task = y_all[:, task_idx]
            mask = ~torch.isnan(y_task)
            if mask.sum() == 0:
                continue

            y_masked = y_task[mask]

            if tc["task_type"] == "regression":
                gamma, nu, alpha, beta = pred
                loss = evidential_regression_loss(
                    gamma[mask], nu[mask], alpha[mask], beta[mask], y_masked
                )
            else:
                logits = pred[mask].squeeze(-1)
                pw = self._pos_weights.get(name)
                if pw is not None:
                    pw = pw.to(logits.device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_masked, pos_weight=pw
                )

            task_losses[name] = loss

        return task_losses

    def training_step(self, batch, batch_idx: int) -> Tensor:
        preds = self(batch)
        task_losses = self._compute_task_losses(preds, batch)

        if not task_losses:
            return torch.tensor(0.0, requires_grad=True)

        # Build ordered loss list aligned with self.task_names
        ordered_losses: list[Tensor] = []
        active_indices: list[int] = []
        for i, name in enumerate(self.task_names):
            if name in task_losses:
                ordered_losses.append(task_losses[name])
                active_indices.append(i)

        # Initialise initial losses on first batch
        if self._initial_losses is None:
            with torch.no_grad():
                self._initial_losses = [
                    task_losses[self.task_names[i]].item()
                    if self.task_names[i] in task_losses
                    else 1.0
                    for i in range(len(self.task_names))
                ]

        # GradNorm weighted total
        # Use subset weights for active tasks
        active_weights_idx = active_indices
        active_initial = [self._initial_losses[i] for i in active_weights_idx]

        # Simple weighted sum using gradnorm weights
        total_loss = self.gradnorm.get_weighted_loss(ordered_losses)

        # Log individual task losses
        for name, loss in task_losses.items():
            self.log(f"train/loss/{name}", loss, on_step=False, on_epoch=True)
        self.log("train/loss/total", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> None:
        preds = self(batch)
        y_all: Tensor = batch.y  # [B, num_tasks]

        for task_idx, tc in enumerate(self.task_configs):
            name = tc["name"]
            pred = preds[name]
            y_task = y_all[:, task_idx]
            mask = ~torch.isnan(y_task)

            if mask.sum() == 0:
                continue

            y_masked = y_task[mask].cpu()

            if tc["task_type"] == "classification":
                prob = torch.sigmoid(pred[mask].squeeze(-1)).cpu()
                self._val_preds[name].append(prob)
            else:
                gamma = pred[0][mask].squeeze(-1).cpu()
                self._val_preds[name].append(gamma)

            self._val_targets[name].append(y_masked)

    def on_validation_epoch_end(self) -> None:
        composite_scores: list[float] = []
        weights: list[float] = []

        for tc in self.task_configs:
            name = tc["name"]
            if not self._val_preds[name]:
                continue

            preds_cat = torch.cat(self._val_preds[name])
            targets_cat = torch.cat(self._val_targets[name])

            if tc["task_type"] == "classification":
                if len(preds_cat) < 2 or targets_cat.unique().numel() < 2:
                    score = 0.5
                else:
                    auroc = BinaryAUROC()
                    score = auroc(preds_cat, targets_cat.long()).item()
                self.log(f"val/auroc/{name}", score)
                composite_scores.append(score)
            else:
                metric = tc.get("metric", "MAE")
                if metric == "MAE":
                    mae_fn = MeanAbsoluteError()
                    mae = mae_fn(preds_cat, targets_cat).item()
                    self.log(f"val/mae/{name}", mae)
                    score = 1.0 / (1.0 + mae)
                elif metric == "Spearman":
                    sp_fn = SpearmanCorrCoef()
                    sp = sp_fn(preds_cat, targets_cat).item()
                    self.log(f"val/spearman/{name}", sp)
                    score = (sp + 1.0) / 2.0
                else:
                    mae_fn = MeanAbsoluteError()
                    mae = mae_fn(preds_cat, targets_cat).item()
                    score = 1.0 / (1.0 + mae)

                composite_scores.append(score)

            weights.append(tc.get("weight", 1.0))

        # Weighted geometric mean
        if composite_scores:
            import numpy as np

            log_sum = sum(
                w * np.log(max(s, 1e-8))
                for s, w in zip(composite_scores, weights)
            )
            composite = np.exp(log_sum / sum(weights))
            self.log("val/composite_score", composite, prog_bar=True)

        # Reset accumulators
        self._val_preds = {n: [] for n in self.task_names}
        self._val_targets = {n: [] for n in self.task_names}

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        param_groups = [
            {"params": self.graph_encoder.parameters(), "lr": self.lr_graph},
            {"params": self.bert_encoder.parameters(), "lr": self.lr_bert},
            {
                "params": list(self.fusion.parameters())
                + list(self.trunk.parameters())
                + list(self.task_heads.parameters())
                + list(self.gradnorm.parameters()),
                "lr": self.lr_rest,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.lr_graph, self.lr_bert, self.lr_rest],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self) -> None:
        """Unfreeze BERT after freeze_epochs warm-up."""
        if self.current_epoch == self.freeze_epochs:
            logger.info("Unfreezing ChemBERTa at epoch %d", self.current_epoch)
            self.bert_encoder.unfreeze()
        elif self.current_epoch < self.freeze_epochs:
            self.bert_encoder.freeze()
