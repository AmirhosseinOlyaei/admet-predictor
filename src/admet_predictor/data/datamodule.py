"""LightningDataModule for ADMET multi-task learning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from admet_predictor.data.dataset import ADMETDataset
from admet_predictor.data.standardize import standardize_smiles

logger = logging.getLogger(__name__)


class ADMETDataModule(LightningDataModule):
    """LightningDataModule that loads processed ADMET parquet files.

    Parameters
    ----------
    data_dir:
        Directory containing per-task parquet files (produced by download_data.py).
    task_configs:
        List of task config dicts from admet_tasks.yaml.
    batch_size:
        Mini-batch size for DataLoaders.
    num_workers:
        Number of DataLoader worker processes.
    """

    def __init__(
        self,
        data_dir: str | Path,
        task_configs: list[dict],
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.task_configs = task_configs
        self.task_names = [tc["name"] for tc in task_configs]
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataset: Optional[ADMETDataset] = None
        self._val_dataset: Optional[ADMETDataset] = None
        self._test_dataset: Optional[ADMETDataset] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_split_data(
        self, split: str
    ) -> tuple[list[str], dict[str, list]]:
        """Load SMILES and labels for a given split across all tasks.

        Returns (smiles_list, labels_dict).
        smiles_list is the union of all molecules from this split.
        labels_dict maps task_name → list of labels aligned with smiles_list.
        """
        # Collect per-task DataFrames for the requested split
        task_dfs: dict[str, pd.DataFrame] = {}
        for tc in self.task_configs:
            parquet = self.data_dir / f"{tc['name']}.parquet"
            if not parquet.exists():
                logger.warning("Parquet not found: %s", parquet)
                continue
            df = pd.read_parquet(parquet)
            if "split" in df.columns:
                split_key = "valid" if split == "val" else split
                df = df[df["split"] == split_key]
            task_dfs[tc["name"]] = df

        if not task_dfs:
            return [], {}

        # Build unified SMILES index (canonical SMILES as key)
        all_smiles_set: dict[str, int] = {}  # canonical → index
        for task_name, df in task_dfs.items():
            smiles_col = "Drug" if "Drug" in df.columns else "smiles"
            for smi in df[smiles_col]:
                canon = standardize_smiles(str(smi))
                if canon and canon not in all_smiles_set:
                    all_smiles_set[canon] = len(all_smiles_set)

        smiles_list = list(all_smiles_set.keys())
        n = len(smiles_list)

        labels: dict[str, list] = {}
        for task_name, df in task_dfs.items():
            smiles_col = "Drug" if "Drug" in df.columns else "smiles"
            label_arr = [float("nan")] * n

            for _, row in df.iterrows():
                canon = standardize_smiles(str(row[smiles_col]))
                if canon and canon in all_smiles_set:
                    idx = all_smiles_set[canon]
                    try:
                        label_arr[idx] = float(row["Y"])
                    except (ValueError, TypeError):
                        pass

            labels[task_name] = label_arr

        return smiles_list, labels

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        cache_base = self.data_dir / "pyg_cache"

        if stage in ("fit", None):
            train_smiles, train_labels = self._load_split_data("train")
            val_smiles, val_labels = self._load_split_data("val")

            self._train_dataset = ADMETDataset(
                smiles_list=train_smiles,
                labels=train_labels,
                task_configs=self.task_configs,
                root=str(cache_base / "train"),
            )
            self._val_dataset = ADMETDataset(
                smiles_list=val_smiles,
                labels=val_labels,
                task_configs=self.task_configs,
                root=str(cache_base / "val"),
            )

        if stage in ("test", None):
            test_smiles, test_labels = self._load_split_data("test")
            self._test_dataset = ADMETDataset(
                smiles_list=test_smiles,
                labels=test_labels,
                task_configs=self.task_configs,
                root=str(cache_base / "test"),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=[],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def compute_pos_weights(self) -> dict[str, torch.Tensor]:
        """Compute pos_weight tensors for imbalanced binary classification tasks.

        Returns dict mapping task_name → scalar tensor (neg_count / pos_count).
        Only includes classification tasks.
        """
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before compute_pos_weights()")

        pos_weights: dict[str, torch.Tensor] = {}

        for task_idx, tc in enumerate(self.task_configs):
            if tc["task_type"] != "classification":
                continue

            y_all = []
            for data in self._train_dataset:
                y_val = data.y[task_idx].item()
                mask_val = data.task_mask[task_idx].item()
                if mask_val and not (y_val != y_val):  # has label, not NaN
                    y_all.append(y_val)

            if not y_all:
                pos_weights[tc["name"]] = torch.tensor(1.0)
                continue

            y_arr = np.array(y_all)
            n_pos = float((y_arr == 1).sum())
            n_neg = float((y_arr == 0).sum())
            weight = n_neg / (n_pos + 1e-6)
            pos_weights[tc["name"]] = torch.tensor(weight, dtype=torch.float32)

        return pos_weights
