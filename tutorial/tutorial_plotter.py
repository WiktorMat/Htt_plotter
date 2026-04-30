"""
Htt_plotter FULL PIPELINE TUTORIAL

THIS IS A COMPLETE USER TUTORIAL.
    It covers FULL workflow:
        1. Merge parquet files
        2. Generate JSON configs (files.json + process.json)
        3. Run plotting pipeline
        4. Render plots in different modes

HOW TO USE
    1. EDIT CONFIG HERE:
        Configurations/config_test/plotter.yaml

        You can change:
        - bins
        - xlim_control / xlim_resolution
        - variables (plotting.control / plotting.resolution)
        - alpha / layout

    2. EDIT DATA LOCATION (if needed)
        In json_generator step below:
            --mc-base data/output/test

    3. RUN:
        python tutorial_plotter.py
"""

import subprocess
import sys
import shlex
from pathlib import Path

CONFIG_NAME = "config_test"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_PATH = PROJECT_ROOT / "source" / "run.py"
JSON_GEN_PATH = PROJECT_ROOT / "scripts" / "json_generator.py"
MERGE_PATH = PROJECT_ROOT / "scripts" / "merge_parquet.py"

def run_command(cmd: str):
    """
    Run command written like in terminal.

    Example:
        python run.py --config config_test --mode raw
    """

    args = shlex.split(cmd)

    # Replace "python script.py" with absolute path
    if args[0] == "python":
        script = args[1]

        if script == "run.py":
            script_path = RUN_PATH
        elif script == "json_generator.py":
            script_path = JSON_GEN_PATH
        elif script == "merge_parquet.py":
            script_path = MERGE_PATH
        else:
            raise ValueError(f"Unknown script: {script}")

        args = [str(script_path)] + args[2:]

    full_cmd = [sys.executable] + args

    print("\n" + "=" * 100)
    print("RUN:", " ".join(full_cmd))
    print("=" * 100)

    subprocess.run(full_cmd, check=True)


def main():
    print("\nHtt_plotter FULL TUTORIAL START\n")

    # STEP 0: OPTIONAL - MERGE PARQUET FILES
    # This step is OPTIONAL.
    # It speeds up processing by merging many small files.
    #
    # ONLY run if:
    # - you have many parquet files per sample
    # - and want faster IO
    #
    # If you get "No parquet files found":
    # - check your data path
    # - or SKIP this step

    DO_MERGE = False

    if DO_MERGE:
        run_command("""
            python merge_parquet.py
            --input data/output/test/Run3_2024/mt/DYto2L
            --output data/output/test/Run3_2024/mt/DYto2L/merged.parquet
    """)

    # STEP 1: GENERATE JSON CONFIGS
    # Creates:
    #   - files.json (list of input files)
    #   - process.json (grouping + colors)

    #USER: change --mc-base to your dataset location
    run_command(f"""
        python json_generator.py
        --config-name {CONFIG_NAME}
        --year Run3_2024
        --channel mt
        --mc-base data/output/test
    """)

    # STEP 2: FULL PIPELINE (RAW MODE)
    # Does EVERYTHING:
    #   - loads data
    #   - applies selections
    #   - fills histograms
    #   - computes MC/Data
    #   - saves parquet

    run_command(f"""
        python run.py
        --config {CONFIG_NAME}
        --mode raw
        --output tutorial_raw
    """)

    # STEP 3: HIST MODE (FAST)
    # Uses saved histograms → NO event loop

    run_command(f"""
        python run.py
        --config {CONFIG_NAME}
        --mode hist
        --output tutorial_hist
    """)

    # STEP 4: PARQUET RENDER MODE
    # Rebuild plots directly from parquet files

    run_command(f"""
        python run.py
        --config {CONFIG_NAME}
        --mode render
        --output tutorial_render
    """)

    # STEP 5: 3D EVENT VISUALIZATION
    # Shows reconstructed physics events in 3D

    run_command(f"""
        python run.py
        --config {CONFIG_NAME}
        --mode 3D_Plot
        --output tutorial_3d
    """)

    print("\nTUTORIAL COMPLETED SUCCESSFULLY\n")
    print("Check results in: plots/tutorial_*")


if __name__ == "__main__":
    main()