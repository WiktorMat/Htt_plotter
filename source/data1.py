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
    def __init__(self, xlim_contrl = None, xlim_resolution = None, bins = None, alpha = None):
        if xlim_contrl is None:
            self.xlim_ctrl = int(input("Set xlim for control: "))
        else:
            self.xlim_ctrl = xlim_contrl
        
        if xlim_resolution is None:
            self.xlim_resol = int(input("Set xlim for resolution: "))
        else:
            self.xlim_resol = xlim_resolution

        self.bins = bins
        self.alpha = alpha
        self.recon = []
        self.contr = []
        self.base_path = Path(__file__).resolve().parent.parent

    def load_data(self, relative_path, secnd_pth):
        full_path = self.base_path / relative_path

        if full_path.exists():
            self.data = pd.read_csv(full_path)

        second_path = self.base_path / secnd_pth

        if second_path.exists():
            self.data2 = pd.read_csv(second_path)



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

        datasets = []
        if hasattr(self, 'data'):
            datasets.append((self.data, self.contr_name, self.recon_name))
        if hasattr(self, 'data2'):
            datasets.append((self.data2, self.contr_name, self.recon_name))

        self.batch_dfs = []

        for data, contr_name, recon_name in datasets:
            contr = data[contr_name]
            recon = data[recon_name]

            resolution = (recon - contr) / contr
            resolution = resolution.replace([np.inf, -np.inf], np.nan).dropna()

            batches = [
                resolution.iloc[i:i+self.batch_size].reset_index(drop=True)
                for i in range(0, len(resolution), self.batch_size)
            ]

            batch_df = pd.concat(batches, axis=1)
            self.batch_dfs.append(batch_df)


    def control_plot(self):
        contr = self.data[self.contr_name]
        contr2 = self.data2[self.contr_name]


        folder_name = "control_plots"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exist.")

        counts1, bins = np.histogram(
            contr,
            bins=self.bins,
            range=(-self.xlim_ctrl, self.xlim_ctrl)
        )

        counts2, bins = np.histogram(
            contr2,
            bins=self.bins,
            range=(-self.xlim_ctrl, self.xlim_ctrl)
        )

        width=np.diff(bins)

        plt.bar(
            bins[:-1],
            counts1,
            width=width,
            edgecolor="black",
            align="edge",
            color = "blue"
        )

        plt.bar(
            bins[:-1],
            counts2,
            width = width,
            bottom=counts1,
            color="lime",
            edgecolor="black",
            align="edge",
            label=f"Control from Z0 - {self.recon_name}"
        )

        file_path = os.path.join(folder_name, f"{self.contr_name}.png")

        plt.title(f"Control {self.contr_name} histogram")
        plt.xlabel(self.contr_name)
        plt.legend([f"Control from Higgs - {self.recon_name}", f"Control from Z0 - {self.recon_name}"], loc="upper right")
        plt.savefig(file_path)
        plt.clf()
        print(f"File saved to {self.contr_name}.png")

    def resolution_plot(self):

        folder_name = "resolution_plots"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exist.")

        counts, bins = np.histogram(
            self.batch_dfs[0],
            bins=self.bins,
            range=(-self.xlim_resol, self.xlim_resol)
        )

        counts2, bins = np.histogram(
            self.batch_dfs[1],
            bins=self.bins,
            range=(-self.xlim_resol, self.xlim_resol)
        )

        plt.bar(
            bins[:-1],
            counts,
            width=np.diff(bins),
            edgecolor="black",
            align="edge"
        )

        plt.bar(
            bins[:-1],
            counts2,
            bottom=counts, 
            color="lime", 
            edgecolor="black", 
            width=np.diff(bins),
            align="edge"
        )

        file_path = os.path.join(folder_name, f"Test_{self.recon_name}_from_{self.contr_name}.png")

        plt.title(f"Resolution {self.recon_name} histogram")
        plt.legend(
            [f"Reconstruction for Higgs - {self.recon_name}", f"Reconstruction for Z0 - {self.recon_name}"],
            loc="upper right")
        plt.savefig(file_path)
        plt.clf()
        print(f"File saved to Test_{self.recon_name}_from_{self.contr_name}.png")

object = Plotter(100, 50, 20, 1)
print(object.load_data("data/Higgs.csv", "data/Z0.csv"))
print(object.set_parameters("trueMETx", "METx"))
print(object.batch(250))
print(object.control_plot())
print(object.resolution_plot())