# -*- coding: utf-8 -*-
"""Pipeline controller for the public release."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MainController:
    def __init__(self, data_root: str, work_dir: str, include_synthetic_in_training: bool = False) -> None:
        self.data_root = os.path.abspath(data_root)
        self.work_dir = os.path.abspath(work_dir)
        self.include_synthetic_in_training = include_synthetic_in_training
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_root = os.path.join(self.work_dir, f"run_{self.timestamp}")
        self.data_dir = os.path.join(self.run_root, "preprocessed")
        self.sample_dir = os.path.join(self.run_root, "samples")
        self.result_dir = os.path.join(self.run_root, "interpolation")
        self.figure_dir = os.path.join(self.run_root, "figures")
        self.script_dir = Path(__file__).resolve().parent
        self.stage_scripts: Dict[str, str] = {
            "preprocess": str(self.script_dir / "data_preprocess.py"),
            "interpolate": str(self.script_dir / "gaussian_interpolation.py"),
            "plot": str(self.script_dir / "result_plot.py"),
        }

    def prepare_dirs(self) -> None:
        for path in [self.run_root, self.data_dir, self.sample_dir, self.result_dir, self.figure_dir]:
            os.makedirs(path, exist_ok=True)

    def run_stage(self, stage: str, cmd: List[str]) -> None:
        logging.info("Running stage: %s", stage)
        logging.info("Command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def write_run_config(self) -> None:
        config = {
            "timestamp": self.timestamp,
            "data_root": self.data_root,
            "run_root": self.run_root,
            "python_executable": sys.executable,
            "include_synthetic_in_training": self.include_synthetic_in_training,
            "stage_scripts": self.stage_scripts,
        }
        with open(os.path.join(self.run_root, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def execute(self) -> None:
        self.prepare_dirs()
        self.write_run_config()

        self.run_stage(
            "preprocess",
            [
                sys.executable,
                self.stage_scripts["preprocess"],
                "--data_root", self.data_root,
                "--output_dir", self.data_dir,
                "--sample_dir", self.sample_dir,
            ],
        )

        interp_cmd = [
            sys.executable,
            self.stage_scripts["interpolate"],
            "--input_csv", os.path.join(self.data_dir, "preprocessed_data.csv"),
            "--output_dir", self.result_dir,
        ]
        if self.include_synthetic_in_training:
            interp_cmd.append("--include_synthetic_in_training")
        self.run_stage("interpolate", interp_cmd)

        self.run_stage(
            "plot",
            [
                sys.executable,
                self.stage_scripts["plot"],
                "--data_dir", self.data_dir,
                "--result_dir", self.result_dir,
                "--output_dir", self.figure_dir,
            ],
        )

        logging.info("Pipeline finished successfully. Outputs in %s", self.run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end controller for the calibration GP pipeline.")
    parser.add_argument("--data_root", required=True, help="Directory containing raw calibration CSV files")
    parser.add_argument("--work_dir", required=True, help="Directory where run outputs will be created")
    parser.add_argument("--include_synthetic_in_training", action="store_true", help="Include synthetic rows in GP training instead of observed-only mode")
    args = parser.parse_args()

    controller = MainController(
        data_root=args.data_root,
        work_dir=args.work_dir,
        include_synthetic_in_training=args.include_synthetic_in_training,
    )
    controller.execute()


if __name__ == "__main__":
    main()
