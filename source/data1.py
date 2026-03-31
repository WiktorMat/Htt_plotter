import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

higgs = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter/data/Higgs.csv")

def funk_bar(wybrane):

    brak = higgs[wybrane]

    print(brak.plot.hist())
    plt.savefig(f"{wybrane}.png")

# print(funk_bar("H.m"))

def sprawdzanie(rekonstrukcja, prawdziwa):

    rek = higgs[rekonstrukcja]
    prawd = higgs[prawdziwa]

    roznica = (rek - prawd)/prawd
    print(roznica.plot.hist(bins=1000, alpha=0.6, xlim=(-35, 35)))
    plt.savefig(f"roznica_{rekonstrukcja}_od_{prawdziwa}")

# print(sprawdzanie("METx", "trueMETx"))

class Plotter:
    def __init__(self, xlim, bins, alpha, kontr_n, rekon_n):
        self.xlim = xlim
        self.bins = bins
        self.alpha = alpha
        self.rekon = rekon_n
        self.kontr = kontr_n

    def load_data(self):
        self.data = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter\data\Higgs.csv")

    def set_parameters(self):
        self.kontr_name = self.kontr
        self.rekon_name = self.rekon

        self.kontr = self.data[self.kontr_name]
        self.rekon = self.data[self.rekon_name]

    def control_plot(self):
        self.kontr.plot.hist()
        plt.savefig(f"{self.kontr_name}.png")
        plt.clf()

    def resolution_plot(self):
        roznica = (self.rekon - self.kontr) / self.kontr
        roznica.plot.hist(range = (-self.xlim, self.xlim), bins = self.bins, alpha = self.alpha)
        plt.savefig(f"sprawdzanie_{self.rekon_name}_od_{self.kontr_name}.png")

object = Plotter(10, 50, 0.7, "trueMETy", "METy")
print(object.load_data())
print(object.set_parameters())
print(object.control_plot())
print(object.resolution_plot())