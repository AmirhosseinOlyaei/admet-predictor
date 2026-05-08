#!/usr/bin/env python
"""Preprocess downloaded ADMET data: standardize SMILES and cache PyG graphs.

Usage
-----
    python scripts/preprocess_data.py --raw-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ADMET raw data")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with downloaded parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for preprocessed parquet files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs/data/admet_tasks.yaml",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        data_config = yaml.safe_load(f)
    task_configs = data_config["tasks"]

    from admet_predictor.data.standardize import standardize_dataframe

    for tc in task_configs:
        task_name = tc["name"]
        raw_path = args.raw_dir / f"{task_name}.parquet"
        out_path = args.output_dir / f"{task_name}.parquet"

        if not raw_path.exists():
            logger.warning("Raw file not found: %s — skipping", raw_path)
            continue

        if out_path.exists():
            logger.info("Already preprocessed: %s", out_path)
            continue

        logger.info("Preprocessing %s ...", task_name)
        df = pd.read_parquet(raw_path)
        smiles_col = "Drug" if "Drug" in df.columns else "smiles"

        original_len = len(df)
        df = standardize_dataframe(df, smiles_col=smiles_col)
        logger.info(
            "%s: %d → %d rows after standardization",
            task_name,
            original_len,
            len(df),
        )

        df.to_parquet(out_path, index=False)
        logger.info("Saved to %s", out_path)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
