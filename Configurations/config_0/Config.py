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

###Selections
PT_1_CUT = 25
PT_2_CUT = 20
ETA_1_CUT = 2.4
ETA_2_CUT = 2.3
DECAYMODE_2_CUT = 1
IDJET_2_CUT = 5
IDE_2_CUT = 2
IDMU_2_CUT = 4
ISO_1_CUT = 0.15
IP_LENSIG_1_CUT = 1

###Plotting
CONTROL = ["trueMETx", "trueMETy", "pt_1", "eta_1"]
RESOL = ["METx", "METy", "pt_2", "eta_2"]


RESOLUTION_PAIRS = [
    ("trueMETx", "METx"),
    ("trueMETy", "METy"),
    ("pt_1", "pt_2"),
    ("eta_1", "eta_2"),
]

SAMPLE_LABELS = {
    "DYto2Mu-2Jets_Bin-0J-MLL-50": "DY → 2μ",
    "DYto2Tau-2Jets_Bin-0J-MLL-50": "DY → 2τ",
    "TTto2L2Nu": "tt̄ dilepton",
    "WtoTauNu-2Jets": "W → τν"
}