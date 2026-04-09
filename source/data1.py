import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
from selection import *
from config import process, BASE_PATH

class Plotter:
    def __init__(self, xlim_contrl=None, xlim_resolution=None, bins=20, alpha=1):
        self.xlim_ctrl = xlim_contrl or 100
        self.xlim_resol = xlim_resolution or 50
        self.bins = bins
        self.alpha = alpha
        self.recon = []
        self.contr = []
        self.project_root = Path(__file__).resolve().parent.parent
        self.all_data = []
        self.event_colors = {
                "Higgs": "blue",
                "Z0": "lime"
            }

    ### Generate JSON automatically from folders in config/process
    def generate_json_from_selection(self, nominal="nominal"):
        files_dict = {"Higgs": [], "Z0": []}
        data_folder = Path(BASE_PATH)

        for proc in process:
            folder = data_folder / proc / nominal
            if not folder.exists():
                print(f"Folder doesn't exists: {folder}")
                continue
            for f in os.listdir(folder):
                if not f.endswith(".parquet"):
                    continue
                full_path = str(folder / f)
                if "Higgs" in f:
                    files_dict["Higgs"].append(full_path)
                elif "Z0" in f or "DYto2" in f or "TT" in f or "Wto" in f:
                    files_dict["Z0"].append(full_path)
                else:
                    files_dict["Z0"].append(full_path)

        project_root = Path(__file__).resolve().parent.parent
        source_folder = project_root / "source"
        source_folder.mkdir(exist_ok=True)

        json_path = source_folder / "files.json"
        with open(json_path, "w") as f:
            json.dump(files_dict, f, indent=4)

        print(f"JSON created at {json_path}")
        return json_path

    ### Loading data files from generated JSON
    def load_data(self, json_path=None):
        self.all_paths = []
        self.all_data = []

        if json_path is None:
            json_path = self.project_root / "source" / "files.json"

        with open(json_path) as f:
            files_dict = json.load(f)

        for event, file_list in files_dict.items():
            for file_path_str in file_list:
                full_path = Path(file_path_str)
                self.all_paths.append(full_path.name)

                if full_path.exists():
                    try:
                        ext = full_path.suffix.lower()
                        if ext == ".csv":
                            df = pd.read_csv(full_path)
                        elif ext == ".parquet":
                            df = pd.read_parquet(full_path)
                        else:
                            print(f"Unsupported file type: {full_path.name}")
                            continue

                        self.all_data.append({
                            "name": full_path.name,
                            "data": df,
                            "event": event
                        })

                        print(f"Loaded: {full_path.name}")

                    except Exception as e:
                        print(f"Error while loading {full_path.name}: {e}")
                else:
                    print(f"File {full_path} does not exist")

    ### Set parameters for control/resolution plotting
    def set_parameters(self, contr_name=None, recon_name=None):
        self.contr_name = []
        self.recon_name = []

        for item in self.all_data:
            df = item["data"]
            cols = plotting(df)
            self.contr_name.extend(cols.get("control", []))
            self.recon_name.extend(cols.get("resolution", []))

        self.contr_name = list(dict.fromkeys(self.contr_name))
        self.recon_name = list(dict.fromkeys(self.recon_name))

        if not self.contr_name and not self.recon_name:
            print("No plotting columns found in any data.")

        all_columns = set(col for item in self.all_data for col in item["data"].columns)
        if contr_name and contr_name in all_columns:
            self.contr_name.append(contr_name)
        if recon_name and recon_name in all_columns:
            self.recon_name.append(recon_name)

        for i, c in enumerate(self.contr_name):
            r = self.recon_name[i] if i < len(self.recon_name) else "N/A"
            print(f"Control file is {c}, and resolution is {r}")

    ### Batch processing
    def batch(self, batch_size=250):
        self.batch_size = batch_size
        self.batch_dfs = {}

        for contr_name, recon_name in zip(self.contr_name, self.recon_name):
            key = f"{contr_name}_{recon_name}"
            self.batch_dfs[key] = []

            for item in self.all_data:
                df = item.get("filtered_data", item["data"])
                file_name = item["name"]

                if contr_name not in df.columns or recon_name not in df.columns:
                    continue

                contr = df[contr_name]
                recon = df[recon_name]
                resolution = (recon - contr) / contr
                resolution = resolution.replace([np.inf, -np.inf], np.nan).dropna()

                batches = [
                    resolution.iloc[j:j+self.batch_size].reset_index(drop=True)
                    for j in range(0, len(resolution), self.batch_size)
                ]

                batch_df = pd.concat(batches, axis=1)

                self.batch_dfs[key].append({
                    "name": file_name,
                    "data": batch_df,
                    "event": item.get("event", "unknown")
                })

    ### Control plotting
    def control_plot(self):
        if len(self.all_data) == 0:
            print("No data in all_data")
            return

        folder_name = "control_plots"
        os.makedirs(folder_name, exist_ok=True)

        json_path = Path(__file__).resolve().parent.parent / "source" / "files.json"
        with open(json_path) as f:
            files_event_map = json.load(f)

        file_to_event = {}
        for event, file_list in files_event_map.items():
            for f in file_list:
                file_to_event[Path(f).name] = event

        for contr_name in self.contr_name:
            plt.figure()
            bottom_counts = None
            for item in self.all_data:
                df = item.get("filtered_data", item["data"])
                if contr_name not in df.columns:
                    continue

                file_name = item["name"]
                event = file_to_event.get(file_name, item.get("event", "unknown")).capitalize()
                color = self.event_colors.get(event, "black")

                contr = df[contr_name]
                counts, bins = np.histogram(
                    contr,
                    bins=self.bins,
                    range=(-self.xlim_ctrl, self.xlim_ctrl)
                )

                if bottom_counts is None:
                    bottom_counts = np.zeros_like(counts)

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom_counts,
                    color=color,
                    edgecolor="black",
                    align="edge",
                )

                bottom_counts += counts

            plt.title(f"Control {contr_name} histogram")
            plt.xlabel(contr_name)
            plt.ylabel("Amount")
            plt.grid(True)
            plt.legend([f"{event} - {contr_name}" for event in self.event_colors.keys()], loc="upper right")
            plt.savefig(os.path.join(folder_name, f"{contr_name}.png"))
            plt.clf()
            print(f"File saved to {folder_name}/{contr_name}.png")

    ### Resolution plotting
    def resolution_plot(self):
        if not hasattr(self, "batch_dfs") or len(self.batch_dfs) == 0:
            print("No batch data available")
            return

        folder_name = "resolution_plots"
        os.makedirs(folder_name, exist_ok=True)

        json_path = Path(__file__).resolve().parent.parent / "source" / "files.json"
        with open(json_path) as f:
            files_event_map = json.load(f)

        file_to_event = {}
        for event, file_list in files_event_map.items():
            for f in file_list:
                file_to_event[Path(f).name] = event

        for key in self.batch_dfs:
            plt.figure()
            bottom_counts = None

            for item in self.batch_dfs[key]:
                batch_df = item["data"]
                file_name = item["name"]
                values = batch_df.values.flatten()
                counts, bins = np.histogram(values, bins=self.bins, range=(-self.xlim_resol, self.xlim_resol))

                event = file_to_event.get(file_name, item.get("event", "unknown")).capitalize()
                color = self.event_colors.get(event, "black")

                if bottom_counts is None:
                    bottom_counts = np.zeros_like(counts)

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom_counts,
                    color=color,
                    edgecolor="black",
                    align="edge",
                )

                bottom_counts += counts

            contr_name, recon_name = key.rsplit("_", 1)
            plt.title(f"Resolution {recon_name} histogram")
            plt.xlabel(recon_name)
            plt.ylabel("Amount")
            plt.grid(True)
            plt.legend([f"{event} - {recon_name}" for event in self.event_colors.keys()], loc="upper right")
            plt.savefig(os.path.join(folder_name, f"Test_{recon_name}_from_{contr_name}.png"))
            plt.clf()
            print(f"File saved to {folder_name}/Test_{recon_name}_from_{contr_name}.png")


plotter = Plotter(100, 50, 20, 1)
json_file = plotter.generate_json_from_selection()
plotter.load_data(json_file)
plotter.set_parameters()
plotter.batch(batch_size=250)
plotter.control_plot()
plotter.resolution_plot()