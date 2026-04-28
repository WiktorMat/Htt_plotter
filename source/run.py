import logging
import sys
import subprocess
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from htt_plotter import Plotter



def main():
    parser = argparse.ArgumentParser(description="Run Htt_plotter pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration name (folder inside Configurations/)",
    )

    args = parser.parse_args()
    config_name = args.config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("plotter.log"),
        ],
    )

    config_dir = project_root / "Configurations" / config_name

    # Ensure config exists
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
    plotter.run_all()


if __name__ == "__main__":
    main()