import argparse
import logging
import sys
import subprocess
import argparse
from pathlib import Path
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from htt_plotter import Plotter
from htt_plotter.core.plotter_render_from_parquet import ParquetRenderer


def main():
    parser = argparse.ArgumentParser(description="Run Htt_plotter pipeline")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=["raw", "hist", "render", "3D_Plot"],
        help="Pipeline mode",
    )

    args = parser.parse_args()
    config_name = args.config

    config_path = project_root / "Configurations" / config_name / "plotter.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    runtime = cfg.get("plotter_runtime", {})

    xlim_control = runtime.get("xlim_control", 100)

    xlim_resolution = runtime.get("xlim_resolution", [-2, 2])
    bin_width_resolution = runtime.get("bin_width_resolution", 0.05)

    if "bin_width_resolution" in runtime:
        res_bins = int((xlim_resolution[1] - xlim_resolution[0]) / bin_width_resolution)
    else:
        res_bins = runtime.get("bins", 20)


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    
    if args.mode in ["raw", "hist", "3D_Plot"]:
        plotter = Plotter(
            xlim_control=xlim_control,
            xlim_resolution=xlim_resolution,
            bins=res_bins,
            alpha=runtime.get("alpha", 0.4),
            layout=runtime.get("layout", "stacked"),
            config_name=config_name,
            mode=args.mode,
            output_suffix=args.output,
        )

        plotter.run_all()
        return

    if args.mode == "render":

        plotter = Plotter(
            xlim_control=xlim_control,
            xlim_resolution=xlim_resolution,
            bins=res_bins,
            alpha=runtime.get("alpha", 0.4),
            layout=runtime.get("layout", "stacked"),
            config_name=config_name,
            mode=args.mode,
            output_suffix=args.output,
        )

        renderer = ParquetRenderer(
            base_dir="plots",
            output_suffix=args.output,
            process_draw_order=plotter._process_draw_order,
            process_colors=plotter.process_colors,
        )

        renderer.run_all()
        return

if __name__ == "__main__":
    main()