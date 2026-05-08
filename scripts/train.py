#!/usr/bin/env python
"""Train the ADMET multi-task model.

Usage
-----
    python scripts/train.py \\
        --config configs/model/attentivefp_base.yaml \\
        --data-dir data/processed \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ADMET multi-task model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/attentivefp_base.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/admet_tasks.yaml",
        help="Path to data/task YAML config",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Directory for checkpoints and logs",
    )
    args = parser.parse_args()

    from admet_predictor.training.train import train

    train(
        config_path=args.config,
        data_config_path=args.data_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
