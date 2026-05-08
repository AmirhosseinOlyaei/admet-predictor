"""PyTorch Geometric dataset for ADMET multi-task learning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

from admet_predictor.data.featurize import mol_to_graph

logger = logging.getLogger(__name__)


class ADMETDataset(InMemoryDataset):
    """In-memory PyG dataset for multi-task ADMET prediction.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings.
    labels:
        Dict mapping task_name → 1-D numpy/tensor array of labels,
        aligned with smiles_list. Use float('nan') for missing labels.
    task_configs:
        List of task config dicts (from admet_tasks.yaml).
    root:
        Root directory for caching processed data.
    transform, pre_transform, pre_filter:
        Standard PyG transform arguments.
    """

    def __init__(
        self,
        smiles_list: list[str],
        labels: dict[str, list],
        task_configs: list[dict],
        root: str = "/tmp/admet_dataset",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        self.smiles_list = smiles_list
        self.labels = labels
        self.task_configs = task_configs
        self.task_names = [tc["name"] for tc in task_configs]
        self._num_tasks = len(task_configs)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["admet_data.pt"]

    def download(self) -> None:
        # Data is provided in-memory, no download needed
        pass

    def process(self) -> None:
        data_list: list[Data] = []

        for idx, smiles in enumerate(self.smiles_list):
            graph = mol_to_graph(smiles)
            if graph is None:
                logger.warning("Skipping invalid SMILES at index %d: %r", idx, smiles)
                continue

            # Build label tensor with NaN for missing values
            y = torch.full((self._num_tasks,), float("nan"), dtype=torch.float32)
            task_mask = torch.zeros(self._num_tasks, dtype=torch.bool)

            for task_idx, task_name in enumerate(self.task_names):
                if task_name in self.labels:
                    task_labels = self.labels[task_name]
                    if idx < len(task_labels):
                        val = task_labels[idx]
                        if val is not None and val == val:  # not NaN
                            y[task_idx] = float(val)
                            task_mask[task_idx] = True

            data = Data(
                x=graph["node_features"],
                edge_index=graph["edge_index"],
                edge_attr=graph["edge_features"],
                y=y,
                task_mask=task_mask,
            )
            data.smiles = smiles

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        logger.info("Processed %d molecules into graph dataset", len(data_list))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        return data
