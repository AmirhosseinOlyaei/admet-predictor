#!/usr/bin/env python
"""Download ADMET datasets from TDC.

Usage
-----
    python scripts/download_data.py --output-dir data/raw
    python scripts/download_data.py --output-dir data/raw --tasks herg ames dili
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ADMET datasets from TDC")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded parquet files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs/data/admet_tasks.yaml",
        help="Path to admet_tasks.yaml",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Subset of task names to download (default: all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        data_config = yaml.safe_load(f)
    task_configs = data_config["tasks"]

    if args.tasks:
        task_names = set(args.tasks)
        task_configs = [tc for tc in task_configs if tc["name"] in task_names]
        if not task_configs:
            logger.error("No matching tasks found for: %s", args.tasks)
            sys.exit(1)

    logger.info("Downloading %d tasks to %s", len(task_configs), args.output_dir)

    from admet_predictor.data.download import download_tdc_tasks

    results = download_tdc_tasks(output_dir=args.output_dir, task_configs=task_configs)

    success = sum(1 for df in results.values() if len(df) > 0)
    logger.info("Downloaded %d/%d tasks successfully", success, len(task_configs))


if __name__ == "__main__":
    main()
