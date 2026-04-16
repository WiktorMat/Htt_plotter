from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DATA = PROJECT_ROOT / "data" / "output" / "test_plotter"

YEAR = "Run3_2024"
CHANNEL = "mt"

COLOR_PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

def get_color(i):
    return COLOR_PALETTE[i % len(COLOR_PALETTE)]


def scan_samples():
    base_path = BASE_DATA / YEAR / CHANNEL

    samples = {}
    idx = 0

    for process_dir in base_path.iterdir():
        if not process_dir.is_dir():
            continue

        sample_name = process_dir.name

        dirs = []

        for syst_dir in process_dir.iterdir():
            if syst_dir.is_dir():
                if list(syst_dir.rglob("*.parquet")):
                    dirs.append(str(syst_dir.as_posix()))

        if not dirs:
            continue

        samples[sample_name] = {
            "kind": "mc",
            "scale": 1.0,
            "color": get_color(idx),
            "dirs": dirs
        }

        idx += 1

    return samples


def add_data(samples):
    samples["Data"] = {
        "kind": "data",
        "color": "black",
        "dirs": [
            str((BASE_DATA / "Muon1").as_posix())
        ]
    }

    return samples


def main():
    samples = scan_samples()
    samples = add_data(samples)

    out_path = PROJECT_ROOT / "Configurations" / "config_0" / "files.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    print(f"Saved files.json → {out_path}")


if __name__ == "__main__":
    main()