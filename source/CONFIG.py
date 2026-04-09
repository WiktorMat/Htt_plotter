###Files .parquet
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_PATH = PROJECT_ROOT / "data" / "Run3_2024" / "mt"

process = [
    "DYto2Mu-2Jets_Bin-0J-MLL-50",
    "DYto2Tau-2Jets_Bin-0J-MLL-50",
    "TTto2L2Nu",
    "WtoTauNu-2Jets"
]
process = ["DYto2Mu-2Jets_Bin-0J-MLL-50", "DYto2Tau-2Jets_Bin-0J-MLL-50", "TTto2L2Nu", "WtoTauNu-2Jets"]