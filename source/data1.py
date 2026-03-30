import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def main(argv: list[str] | None = None) -> int:
#     parser = argparse.ArgumentParser(description=)

#     default_csv = Path(__file__).resolve().parents[1] / "data" / "Higgs.csv"
#     file_pth = Path(args.file) if args.file is not None else default_csv

df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

# titanic = df["Age"]
# titanic = df.loc[df["Age"] > 23]
# print(titanic)

#każdy wiek wypisanie
# print(df["Age"])

#maksymalny wiek wypisanie
# print(df["Age"].max())

#opis Wieku
#print(df.describe())

# Wyciąganie plików z data:
# higgs = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter/data/Higgs.csv")

#pierwsze n wierszy wyświetlanie (tail ostatnie n wierszy)
# print(higgs.head(8))

# Sprawdzanie typów danych
# print(higgs.dtypes)

# Techniczne informacje o kodzie:
# print(higgs.info())

# Tworzenie exela:
# print(higgs.to_excel("higs.xlxs", sheet_name="dane", index=False))
# print(higgs = pd.read_exel("higs.xlxs", sheet_name="dane"))

# Pojedyńcze dane wyciąganie
# ages = higgs["METx"]
# print(ages.head())

# Sprawdzanie "Series" danych
# print(type(higgs["METx"]))

# Ilość danych w 1 kolumnie
# print(higgs["METx"].shape)

# zmienna mająca więcej niż 1 danę
# do_zmiennych = higgs[["METx", "eta1"]]

# wypisywanie wielu column
# print(do_zmiennych.head(3))

# typ danych w jakim są:
# print(type(higgs[["METx", "eta1"]]))

# Ilość danych z wielu kolumn
# print(do_zmiennych.shape)

# # Wyświetlanie interesujących wierszy (wiersze mające w METx więcej niż 15)
# pow_25 = do_zmiennych[higgs[["METx"]] > 25]
# print(pow_25)

# Sprawdzanie większe-mniejsze
# print(higgs[["METx"]] > 25)

#Ilość powyżej-poniżej
# print(pow_25.shape)

# Zwracanie prawdziwych
# pon_200 = higgs[higgs["METx"].isin([0, 200])]
# print(pon_200.head())

# OR
# dane_200 = hig[(higgs["METx"] == 22.200546) | (higgs["METx"] == 8.349314)]
# print(dane_200.head)

# Posiadane dane
# brak = higgs[higgs["METx"].notna()]
# print(brak.head())
# print(brak.shape)

# Wyświetl coś gdzie inna dana jest większa/mniejsza/równa
# wieksz_200 = higgs.loc[higgs["covYY"] > 125, "H.m"]
# print(wieksz_200.head())

# Wyświetl wiersze od do i kolumny od do
# print(higgs.iloc[9:26, 2:5])

# # Przypisywanie innych danych
# higgs.iloc[0:3, 3] = 3
# print(higgs.iloc[:5, 3])


# air_quality
# air_quality = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter\data/air_quality_no2.csv", index_col=0, parse_dates=True)
# print(air_quality.head())

# Tworzenie plot:
# print(air_quality.plot())
# figura = air_quality.plot()
# plt.savefig("obraz.png")


# plt.figure()
# df.plot(kind="bar")

# print(higgs.plot())
# figura = higgs.plot()
# plt.savefig("test.png")

def funk_bar(wybrane):
    higgs = pd.read_csv("D:\Praktyki_zawodowe\Htt_plotter/data/Higgs.csv")

    brak = higgs[wybrane]

    print(brak.plot.hist())
    plt.savefig(f"{wybrane}.png")

print(funk_bar("H.m"))