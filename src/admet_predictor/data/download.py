"""Download ADMET datasets from Therapeutics Data Commons (TDC)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def download_tdc_tasks(
    output_dir: Path,
    task_configs: list[dict],
) -> dict[str, pd.DataFrame]:
    """Download all ADMET tasks from TDC and save as parquet files.

    Parameters
    ----------
    output_dir:
        Directory to save parquet files.
    task_configs:
        List of task config dicts from admet_tasks.yaml.

    Returns
    -------
    Dict mapping task_name → combined DataFrame with columns:
        Drug_ID, Drug (SMILES), Y (label), split
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for task_cfg in task_configs:
        task_name = task_cfg["name"]
        tdc_name = task_cfg["tdc_name"]
        tdc_group = task_cfg["tdc_group"]

        parquet_path = output_dir / f"{task_name}.parquet"

        if parquet_path.exists():
            logger.info("Task %s already cached at %s", task_name, parquet_path)
            results[task_name] = pd.read_parquet(parquet_path)
            continue

        logger.info("Downloading %s (%s) from TDC ...", task_name, tdc_name)
        try:
            if tdc_group == "ADME":
                from tdc.single_pred import ADME  # type: ignore

                dataset = ADME(name=tdc_name)
            else:  # Tox
                from tdc.single_pred import Tox  # type: ignore

                dataset = Tox(name=tdc_name)

            split = dataset.get_split()  # returns dict with 'train', 'valid', 'test'

            dfs = []
            for split_name, df in split.items():
                df = df.copy()
                df["split"] = split_name
                dfs.append(df)

            combined = pd.concat(dfs, ignore_index=True)

            # Normalise column names: TDC uses 'Drug' for SMILES and 'Y' for labels
            if "Drug" not in combined.columns and "smiles" in combined.columns:
                combined = combined.rename(columns={"smiles": "Drug"})
            if "Y" not in combined.columns and "label" in combined.columns:
                combined = combined.rename(columns={"label": "Y"})

            combined.to_parquet(parquet_path, index=False)
            results[task_name] = combined
            logger.info(
                "Saved %d rows for task %s", len(combined), task_name
            )

        except Exception as exc:
            logger.error("Failed to download task %s: %s", task_name, exc)
            # Return empty DataFrame so pipeline can continue
            results[task_name] = pd.DataFrame(
                columns=["Drug_ID", "Drug", "Y", "split"]
            )

    return results
