###Files
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_PATH = PROJECT_ROOT / "data"/ "output" / "test_plotter" / "Run3_2024" / "mt"

process = [
    "DYto2Mu-2Jets_Bin-0J-MLL-50",
    "DYto2Tau-2Jets_Bin-0J-MLL-50",
    "TTto2L2Nu",
    "WtoTauNu-2Jets"
]

QCD = {
    "method": "abcd",
    "regions": {
        "iso": {
            "idDeepTau2018v2p5VSjet_2": [(">=", 5)],
        },
        "antiiso": {
            "idDeepTau2018v2p5VSjet_2": [(">", 1), ("<", 5)],
        },
    },
}

SELECTION = {
    "pt_1": (">", 26),
    "pt_2": (">", 25),
    "eta_1": ("abs<", 2.4),
    "eta_2": ("abs<", 2.5),
    "ip_z_1": ("<", 0.2),
    "ip_z_2": ("<", 0.2),
    "iso_1": ("<", 0.15),
    "idDeepTau2018v2p5VSe_2": (">=", 2),
    "idDeepTau2018v2p5VSmu_2": (">=", 4),
    "n_bjets": ("==", 2),
    # "mt_1": (">", 80),
}