import logging
import sys
import subprocess
import argparse
from pathlib import Path

from htt_plotter import Plotter

project_root = Path(__file__).resolve().parent.parent

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run HTT plotter")
    parser.add_argument("--config-name", default="config_0", help="Configuration folder name")
    parser.add_argument(
        "--backend",
        default="multiprocessing",
        choices=["serial", "multiprocessing", "mp"],
        help="Processing backend for raw mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of processes for multiprocessing backend",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("plotter.log"),
        ],
    )

    config_name = args.config_name

    config_dir = project_root / "Configurations" / config_name

    # Ensure we have an explicit per-config file list for performance.
    # (We still allow source/files.json as a fallback via the config loader.)
    if not (config_dir / "files.json").exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "htt_plotter.tools.json_generator",
                "--config-name",
                config_name,
            ],
            cwd=project_root / "source",
            check=True,
        )

    plotter = Plotter(
        xlim_control=100,
        xlim_resolution=50,
        bins=20,
        alpha=0.4,
        layout="stacked",
        config_name=config_name,
        mode="raw",
    )

    runtime_cfg = plotter.plotter_config.setdefault("plotter_runtime", {})
    runtime_cfg["processing_backend"] = "multiprocessing" if args.backend == "mp" else args.backend
    runtime_cfg["n_workers"] = max(1, int(args.workers))

    plotter.run_all(do_control=False, do_resolution=False, do_mc_data=True)