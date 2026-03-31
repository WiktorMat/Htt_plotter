import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    def __init__(self, xlim, bins, alpha):
        self.xlim = xlim
        self.bins = bins
        self.alpha = alpha
        self.recon = []
        self.contr = []
        self.base_path = Path(__file__).resolve().parent.parent

    def load_data(self, relative_path):
        full_path = self.base_path / relative_path
        
        if full_path.exists():
            self.data = pd.read_csv(full_path)

    def set_parameters(self, contr_name=None, recon_name=None):
        if contr_name is None:
            contr_name = input("Set control: ")
        if recon_name is None:
            recon_name = input("Set reconstruction: ")
        
        self.contr_name = contr_name
        self.recon_name = recon_name
        self.resolution_true = contr_name

    def control_plot(self):
        contr = self.data[self.contr_name]

        counts, bins = np.histogram(
            contr,
            bins=self.bins,
            range=(-self.xlim, self.xlim)
        )

        plt.bar(
            bins[:-1],
            counts,
            width=np.diff(bins),
            edgecolor="black",
            align="edge"
        )

        plt.title(f"Control {self.contr_name} histogram")
        plt.xlabel(self.contr_name)

        plt.savefig(f"{self.contr_name}.png")
        plt.clf()

    def resolution_plot(self):
        resolution_true = self.data[self.resolution_true]
        recon = self.data[self.recon_name]

        resolution = (recon - resolution_true) / resolution_true

        counts, bins = np.histogram(
            resolution,
            bins=self.bins,
            range=(-self.xlim, self.xlim)
        )

        plt.bar(
            bins[:-1],
            counts,
            width=np.diff(bins),
            edgecolor="black",
            align="edge"
        )

        plt.title(f"Resolution {self.recon_name} histogram")
        plt.legend([f"Control - {self.resolution_true}", f"Reconstruction - {self.recon_name}"], loc="upper right")
        plt.savefig(f"Test_{self.recon_name}_from_{self.resolution_true}.png")
        plt.clf()

object = Plotter(20, 20, 1)
print(object.load_data("data/Higgs.csv"))
print(object.set_parameters("trueMETx", "METx"))
print(object.control_plot())
print(object.resolution_plot())