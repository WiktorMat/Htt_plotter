"""
Htt_plotter FULL TUTORIAL SCRIPT

THIS IS A USER-FACING TUTORIAL.

HOW TO USE THIS TUTORIAL
    1. ALL CONFIGURATION IS DONE IN:

    Configurations/config_test/plotter.yaml

    You can change:

    - HISTOGRAM SETTINGS:
            bins:
                -> number of histogram bins (e.g. 10, 20, 80)
                -> affects resolution of ALL plots
            bin_width_resolution:
                -> alternative way to control resolution plots
            xlim_control:
                -> x-axis range for control plots
            xlim_resolution:
                -> x-axis range for resolution plots
            alpha:
                -> transparency of histograms
            layout:
                -> stacking style ("stacked" or "overlay")

        - VARIABLES TO PLOT:
            plotting:
                control:
                -> list of variables for control plots
                    example: trueMETx, trueMETy

                resolution:
                -> pairs [truth, reco]
                    example: [trueMETx, METx]

    - VISUAL OUTPUT:
            extra_plots:
                enable:
                -> enable additional physics plots (advanced)

    2. HOW TO EXPERIMENT

        - To test different binning:
        change "bins" in YAML and rerun script

        - To change plotted variables:
        edit "plotting.control" and "plotting.resolution"

        - To change plot appearance:
        modify:
            - alpha
            - layout
            - xlim_control / xlim_resolution

        - To disable heavy features:
        set:
            extra_plots.enable: false

    3. WHAT THIS SCRIPT DOES

        - runs full reconstruction pipeline (raw mode)
        - builds histograms from events
        - produces MC/Data comparison plots
        - saves parquet intermediate results
        - renders plots from multiple modes:
            * raw (full processing)
            * hist (fast replot)
            * render (parquet-based)
            * 3D event visualization
"""

import subprocess
import sys
from pathlib import Path

CONFIG_NAME = "config_test"
RUN_PATH = Path(__file__).resolve().parents[1] / "source" / "run.py"


def run_command(cmd_args):
    """
    Executes pipeline step.

    IMPORTANT:
    - DO NOT pass full shell commands here
    - ONLY pass argparse arguments (without "python run.py")
    """
    if isinstance(cmd_args, str):
        cmd_args = cmd_args.split()

    full_cmd = [sys.executable, str(RUN_PATH)] + cmd_args

    print("\n" + "=" * 100)
    print("RUN:", " ".join(full_cmd))
    print("=" * 100)

    subprocess.run(full_cmd, check=True)


def main():
    print("\nHtt_plotter TUTORIAL START\n")

    # STEP 1: FULL PIPELINE (RAW MODE)

    # This is the MAIN run:
    # - reads input data
    # - applies selections
    # - fills histograms
    # - computes MC/Data comparison
    # - saves intermediate parquet files
    run_command([
        "--config", CONFIG_NAME,
        "--mode", "raw",
        "--output", "tutorial_raw"
    ])

    # STEP 2: HIST MODE (FAST RE-PLOTTING)

    # Uses already computed histograms
    # DOES NOT rerun event processing
    run_command([
        "--config", CONFIG_NAME,
        "--mode", "hist",
        "--output", "tutorial_hist"
    ])

    # STEP 3: PARQUET RENDER MODE

    # Rebuilds plots directly from saved parquet files
    run_command([
        "--config", CONFIG_NAME,
        "--mode", "render",
        "--output", "tutorial_render"
    ])

    # STEP 4: 3D EVENT VISUALIZATION

    # Shows reconstructed physics events in 3D
    run_command([
        "--config", CONFIG_NAME,
        "--mode", "3D_Plot",
        "--output", "tutorial_3d"
    ])

    print("\nTUTORIAL COMPLETED SUCCESSFULLY\n")
    print("Check output folder: plots/tutorial_*")


if __name__ == "__main__":
    main()