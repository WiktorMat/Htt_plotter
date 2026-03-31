import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)

    def set_parameters(self, contr_name=None, recon_name=None):
        if contr_name is None:
            contr_name = input("Set control: ")
        if recon_name is None:
            recon_name = input("Set reconstruction: ")
        
        self.contr_name = contr_name
        self.recon_name = recon_name

    def control_plot(self):
        contr = self.data[self.contr_name]
        contr.plot.hist(title = f"Control {self.contr_name} histogram", range = (-self.xlim, self.xlim), bins = self.bins, edgecolor="black")
        plt.xlabel(self.contr_name)
        plt.savefig(f"{self.contr_name}.png")
        plt.clf()

    def resolution_plot(self):
        contr = self.data[self.contr_name]
        recon = self.data[self.recon_name]
        roznica = (recon - contr) / contr
        roznica.plot.hist(title = f"Resolution {self.recon_name} histogram", range = (-self.xlim, self.xlim), bins = self.bins, alpha = 1, edgecolor="black")
        plt.legend([f"Control - {self.contr_name}", f"Reconstruction - {self.recon_name}"], loc="upper right")
        plt.savefig(f"Test_{self.recon_name}_od_{self.contr_name}.png")
        plt.clf()

object = Plotter(20, 50, 0.7)
print(object.load_data("D:\Praktyki_zawodowe\Htt_plotter/data/Higgs.csv"))
print(object.set_parameters())
print(object.control_plot())
print(object.resolution_plot())