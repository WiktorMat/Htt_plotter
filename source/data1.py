import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

higgs = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter/data/Higgs.csv")

# def funk_bar(wybrane):

#     brak = higgs[wybrane]

#     print(brak.plot.hist())
#     plt.savefig(f"{wybrane}.png")

# print(funk_bar("H.m"))

# def sprawdzanie(rekonstrukcja, prawdziwa):

#     rek = higgs[rekonstrukcja]
#     prawd = higgs[prawdziwa]

#     roznica = (rek - prawd)/prawd
#     print(roznica.plot.hist(bins=1000, alpha=0.6, xlim=(-35, 35)))
#     plt.savefig(f"roznica_{rekonstrukcja}_od_{prawdziwa}")

# print(sprawdzanie("METx", "trueMETx"))

class Plotter:
    def __init__(self, xlim_contrl=None, xlim_resolution=None, bins=None, alpha=None):
        self.xlim_ctrl = xlim_contrl or int(input("Set xlim for control: "))
        self.xlim_resol = xlim_resolution or int(input("Set xlim for resolution: "))
        self.bins = bins
        self.alpha = alpha
        self.recon = []
        self.contr = []
        self.base_path = Path(__file__).resolve().parent.parent
        self.all_data = []

    def load_data(self, *relative_paths):
        self.all_paths = []
        self.all_data = []

        if len(relative_paths) == 0:
            relative_paths_list = []
            while True:
                new = input("Add new path (end to finish): ")
                if new.lower() == "end":
                    break
                relative_paths_list.append("data\\" +new)
            relative_paths = relative_paths_list

        for rel_path in relative_paths:
            full_path = self.base_path / rel_path
            self.all_paths.append(rel_path)

            if full_path.exists():
                try:
                    df = pd.read_csv(full_path)
                    self.all_data.append({"name": full_path.name, "data": df})
                    print(f"Loaded: {full_path.name}")
                except Exception as e:
                    print(f"Error while loading {full_path.name}: {e}")
            else:
                print(f"File {full_path} does not exist")
            



    def set_parameters(self, contr_name=None, recon_name=None):
        if contr_name is None:
            contr_name = input("Set control: ")
        if recon_name is None:
            recon_name = input("Set reconstruction: ")
        
        self.contr_name = contr_name
        self.recon_name = recon_name

        print(f"Control file is {contr_name}, and resolution is {recon_name}")

    def batch(self, batch_size=None):
        if batch_size is None:
            self.batch_size = int(input("Set batch size: "))
        else:
            self.batch_size = batch_size

        self.batch_dfs = []

        for item in self.all_data:
            df = item["data"]

            if self.contr_name in df.columns and self.recon_name in df.columns:
                contr = df[self.contr_name]
                recon = df[self.recon_name]

                resolution = (recon - contr) / contr
                resolution = resolution.replace([np.inf, -np.inf], np.nan).dropna()

                batches = [
                    resolution.iloc[i:i+self.batch_size].reset_index(drop=True)
                    for i in range(0, len(resolution), self.batch_size)
                ]

                batch_df = pd.concat(batches, axis=1)
                self.batch_dfs.append(batch_df)
            else:
                print(f"Column {self.contr_name} or {self.recon_name} does not exist")


    def control_plot(self):
        if len(self.all_data) == 0:
            print("No data in all_data")
            return

        folder_name = "control_plots"
        os.makedirs(folder_name, exist_ok=True)

        colors = ["blue", "lime", "red", "orange", "purple", "cyan", "magenta"]

        plt.figure()
        bottom_counts = None
        labels = []

        for i, item in enumerate(self.all_data):
            df = item["data"]
            file_name = item["name"]
            if self.contr_name not in df.columns:
                print(f"File {file_name} does not contain {self.contr_name}, next")
                continue

            contr = df[self.contr_name]

            counts, bins = np.histogram(
                contr,
                bins=self.bins,
                range=(-self.xlim_ctrl, self.xlim_ctrl)
            )

            width = np.diff(bins)

            if bottom_counts is None:
                bottom_counts = np.zeros_like(counts)

            plt.bar(
                bins[:-1],
                counts,
                width=width,
                bottom=bottom_counts,
                color=colors[i % len(colors)],
                edgecolor="black",
                align="edge",
            )

            bottom_counts += counts
            labels.append(f"Control for {file_name} - {self.recon_name}")

        plt.title(f"Control {self.contr_name} histogram")
        plt.xlabel(self.contr_name)
        plt.legend(labels, loc="upper right")

        file_path = os.path.join(folder_name, f"{self.contr_name}.png")
        plt.savefig(file_path)
        plt.clf()
        print(f"File saved to {file_path}")

    def resolution_plot(self):

        if not hasattr(self, "batch_dfs") or len(self.batch_dfs) == 0:
            print("No batches for histogram")
            return

        folder_name = "resolution_plots"
        os.makedirs(folder_name, exist_ok=True)

        colors = ["blue", "lime", "red", "orange", "purple", "cyan", "magenta"]

        plt.figure()

        bottom_counts = None

        for i, batch_df in enumerate(self.batch_dfs):
            if hasattr(batch_df, "values"):
                values = batch_df.values.flatten()
            else:
                values = batch_df

            counts, bins = np.histogram(
                values,
                bins=self.bins,
                range=(-self.xlim_resol, self.xlim_resol)
            )

            width = np.diff(bins)

            if bottom_counts is None:
                bottom_counts = np.zeros_like(counts)


            plt.bar(
                bins[:-1],
                counts,
                width=width,
                bottom=bottom_counts,
                color=colors[i % len(colors)],
                edgecolor="black",
                align="edge",
            )

            bottom_counts += counts

        plt.title(f"Resolution {self.recon_name} histogram")
        plt.xlabel(self.recon_name)
        y = 0
        labels = []
        for y, path in enumerate(self.all_paths):
            file_name = Path(path).name
            labels.append(f"Reconstruction for {file_name} - {self.recon_name}")
            y+=1
        plt.legend(labels, loc="upper right")

        file_path = os.path.join(folder_name, f"Test_{self.recon_name}_from_{self.contr_name}.png")
        plt.savefig(file_path)
        plt.clf()
        print(f"File saved to {file_path}")

object = Plotter(100, 50, 20, 1)
print(object.load_data())
print(object.set_parameters("trueMETx", "METx"))
print(object.batch(250))
print(object.control_plot())
print(object.resolution_plot())