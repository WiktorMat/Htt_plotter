import sys
import subprocess
from pathlib import Path

from Plotter import Plotter

project_root = Path(__file__).resolve().parent.parent

#Main
if __name__ == "__main__":

    subprocess.run(
        [sys.executable, "json_generator.py"],
        cwd=project_root / "source",
        check=True
    )

    plotter = Plotter(100, 50, 20, 1)
    plotter.load_index()
    plotter.apply_selection()
    plotter.set_parameters()
    plotter.control_plot()
    plotter.batch()
    plotter.resolution_plot()
    plotter.Plot_MC_Data_Agrement()