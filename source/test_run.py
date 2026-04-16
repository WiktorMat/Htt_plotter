import sys
import subprocess
from pathlib import Path

from Plotter import Plotter

project_root = Path(__file__).resolve().parent.parent

if __name__ == "__main__":

    config_name = "config_0"

    subprocess.run(
        [sys.executable, "json_generator.py"],
        cwd=project_root / "source",
        check=True
    )

    plotter = Plotter(
        100,
        50,
        20,
        1,
        config_name=config_name
    )

    plotter.load_index()
    plotter.apply_selection()
    plotter.set_parameters()
    
    plotter.batch()
    
    for item in plotter.files_index:
        plotter.process_file(item)

    plotter.control_plot()
    plotter.resolution_plot()
    plotter.Plot_MC_Data_Agrement()