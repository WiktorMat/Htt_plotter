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

    def batch(self, batch_size):
        contr = self.data[self.contr_name]
        recon = self.data[self.recon_name]

        if batch_size is None:
            batch_size = int(input("Set batch size: "))

        resolution = (recon - contr) / contr
        resolution = resolution.replace([np.inf, -np.inf], np.nan).dropna()

        batches = [
        resolution.iloc[i:i+batch_size].reset_index(drop=True)
        for i in range(0, len(resolution), batch_size)
        ]

        self.batch_df = pd.concat(batches, axis=1)


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

        counts, bins = np.histogram(
            self.batch_df,
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
print(object.batch(1000))
print(object.control_plot())
print(object.resolution_plot())