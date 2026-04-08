import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# higgs = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter/data/Higgs.csv")

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

        posible_data = []
        folder_path = self.base_path / "data"
        posible_data = os.listdir(folder_path)
        print(posible_data)


        if len(relative_paths) == 0:
            relative_paths_list = []
            while True:
                new = input("Add new path (end to finish): ")
                if new.lower() == "end":
                    break
                if new in posible_data:
                    relative_paths_list.append("data\\" +new)
                else:
                    print("Wrong path, give another")
            relative_paths = relative_paths_list

        for rel_path in relative_paths:
            full_path = self.base_path / "data\\" / rel_path
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
        self.contr_name = []
        self.recon_name = []

        all_columns = set()
        for item in self.all_data:
            df = item["data"]
            all_columns.update(df.columns)

        if contr_name is None or recon_name is None:
            while True:
                contr_name_input = input("Set control: ")
                recon_name_input = input("Set reconstruction: ")

                if contr_name_input not in all_columns:
                    print(f"Column {contr_name_input} does not exist in any file")
                    continue

                if recon_name_input not in all_columns:
                    print(f"Column {recon_name_input} does not exist in any file")
                    continue

                self.contr_name.append(contr_name_input)
                self.recon_name.append(recon_name_input)

                print(f"Added: Control = {contr_name_input}, Resolution = {recon_name_input}")

                end = input("Write 'end' to finish, or press ENTER to add more: ")
                if end.lower() == "end":
                    break
        else:
            if contr_name in all_columns and recon_name in all_columns:
                self.contr_name.append(contr_name)
                self.recon_name.append(recon_name)
            else:
                print("Given columns do not exist!")

        for i, c in enumerate(self.contr_name):
            print(f"Control file is {c}, and resolution is {self.recon_name[i]}")


    def batch(self, batch_size=None):
        if batch_size is None:
            self.batch_size = int(input("Set batch size: "))
        else:
            self.batch_size = batch_size

        self.batch_dfs = {}

        for contr_name, recon_name in zip(self.contr_name, self.recon_name):
            key = f"{contr_name}_{recon_name}"
            self.batch_dfs[key] = []

            for item in self.all_data:
                df = item["data"]
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
                    "data": batch_df
                })


    def control_plot(self):
        if len(self.all_data) == 0:
            print("No data in all_data")
            return

        folder_name = "control_plots"
        os.makedirs(folder_name, exist_ok=True)

        colors = ["blue", "lime", "red", "orange", "purple", "cyan", "magenta"]

        for contr_name in self.contr_name:
            plt.figure()
            bottom_counts = None
            labels = []

            for i, item in enumerate(self.all_data):
                df = item["data"]
                file_name = item["name"]

                if contr_name not in df.columns:
                    print(f"File {file_name} does not contain {contr_name}, skipping")
                    continue

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
                    color=colors[i % len(colors)],
                    edgecolor="black",
                    align="edge",
                )

                bottom_counts += counts
                labels.append(file_name)

            plt.title(f"Control {contr_name} histogram")
            plt.xlabel(contr_name)
            plt.legend([f"From {label} - {contr_name}" for label in labels], loc="upper right")

            file_path = os.path.join(folder_name, f"{contr_name}.png")
            plt.savefig(file_path)
            plt.clf()
            print(f"File saved to {file_path}")

    def resolution_plot(self):
        if not hasattr(self, "batch_dfs") or len(self.batch_dfs) == 0:
            print("No batch data available")
            return

        folder_name = "resolution_plots"
        os.makedirs(folder_name, exist_ok=True)

        colors = ["blue", "lime", "red", "orange", "purple", "cyan", "magenta"]

        for key in self.batch_dfs:
            plt.figure()
            bottom_counts = None
            labels = []

            for i, item in enumerate(self.batch_dfs[key]):
                file_name = item["name"]
                batch_df = item["data"]

                values = batch_df.values.flatten()

                counts, bins = np.histogram(
                    values,
                    bins=self.bins,
                    range=(-self.xlim_resol, self.xlim_resol)
                )

                if bottom_counts is None:
                    bottom_counts = np.zeros_like(counts)

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom_counts,
                    color=colors[i % len(colors)],
                    edgecolor="black",
                    align="edge",
                )

                bottom_counts += counts
                labels.append(file_name)

            contr_name, recon_name = key.split("_")

            plt.title(f"Resolution {recon_name} histogram")
            plt.xlabel(recon_name)
            plt.legend(
                [f"From {l} - {recon_name}" for l in labels],
                loc="upper right"
            )
            file_path = os.path.join(
                folder_name,
                f"Test_{recon_name}_from_{contr_name}.png"
            )
            plt.savefig(file_path)
            plt.clf()
            print(f"File saved to {file_path}")

object = Plotter(100, 50, 20, 1)
print(object.load_data("Higgs.csv", "Z0.csv"))
print(object.set_parameters())
print(object.batch(250))
print(object.control_plot())
print(object.resolution_plot())