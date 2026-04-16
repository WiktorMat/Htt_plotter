from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


def get_color(i: int) -> str:
    return COLOR_PALETTE[i % len(COLOR_PALETTE)]


def scan_samples(base_data: Path = BASE_DATA, year: str = YEAR, channel: str = CHANNEL) -> dict:
    base_path = base_data / year / channel

    samples: dict[str, dict] = {}
    idx = 0

    if not base_path.exists():
        return samples

    for process_dir in base_path.iterdir():
        if not process_dir.is_dir():
            continue

        sample_name = process_dir.name
        dirs: list[str] = []

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
            "dirs": dirs,
        }

        idx += 1

    return samples


def add_data(samples: dict, base_data: Path = BASE_DATA) -> dict:
    samples["Data"] = {
        "kind": "data",
        "color": "black",
        "dirs": [
            str((base_data / "Muon1").as_posix()),
        ],
    }

    return samples


def write_files_json(samples: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate files.json for a plotter config")
    parser.add_argument("--config-name", default="config_0")
    parser.add_argument("--year", default=YEAR)
    parser.add_argument("--channel", default=CHANNEL)
    args = parser.parse_args(argv)

    samples = scan_samples(year=args.year, channel=args.channel)
    samples = add_data(samples)

    out_path = PROJECT_ROOT / "Configurations" / args.config_name / "files.json"
    write_files_json(samples, out_path)

    print(f"Saved files.json → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
