import logging
import sys
import subprocess
from pathlib import Path

from htt_plotter import Plotter

project_root = Path(__file__).resolve().parent.parent

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("plotter.log"),
        ],
    )

    config_name = "config_csv"

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
        xlim_contrl=100,
        xlim_resolution=50,
        bins=20,
        alpha=0.4,
        layout="stacked",
        config_name=config_name,
        mode="raw",
    )

    plotter.run_all()