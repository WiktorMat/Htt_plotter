from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_ROOT = PROJECT_ROOT / "output"
MC_BASE_DEFAULT = OUTPUT_ROOT / "test_plotter"
DATA_DIR_DEFAULT = OUTPUT_ROOT / "Muon1"

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


def _rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def scan_mc_samples(base_dir: Path, year: str = YEAR, channel: str = CHANNEL) -> dict:
    """Scan MC folders and return samples with explicit file lists."""

    base_path = base_dir / year / channel

    print(f"Scanning MC base: {base_path}")

    samples: dict[str, dict] = {}
    idx = 0

    if not base_path.exists():
        print("MC base does not exist; no MC samples found")
        return samples

    for process_dir in sorted([p for p in base_path.iterdir() if p.is_dir()]):
        sample_name = process_dir.name

        files = sorted({p.resolve() for p in process_dir.rglob("*.parquet") if p.is_file()})
        if not files:
            continue

        print(f"  MC sample: {sample_name} | files: {len(files)}")

        samples[sample_name] = {
            "kind": "mc",
            "scale": 1.0,
            "color": get_color(idx),
            "files": [_rel(p) for p in files],
        }

        idx += 1

    return samples


def add_data(samples: dict, data_dir: Path) -> dict:
    print(f"Scanning Data dir: {data_dir}")

    files = sorted({p.resolve() for p in data_dir.rglob("*.parquet") if p.is_file()})
    if not files:
        print("No data parquet files found")
        return samples

    print(f"  Data: files: {len(files)}")

    samples["Data"] = {
        "kind": "data",
        "color": "black",
        "files": [_rel(p) for p in files],
    }

    return samples


def write_files_json(samples: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for cfg in (samples or {}).values():
        total_files += len(cfg.get("files", []) or [])

    print(f"Writing files.json: {out_path} | samples={len(samples)} | total_files={total_files}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate files.json for a plotter config")
    parser.add_argument("--config-name", default="config_0")
    parser.add_argument("--year", default=YEAR)
    parser.add_argument("--channel", default=CHANNEL)
    parser.add_argument(
        "--mc-base",
        default=str(MC_BASE_DEFAULT),
        help="Base folder containing MC samples (default: output/test_plotter)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR_DEFAULT),
        help="Folder containing data parquet files (default: output/Muon1)",
    )
    args = parser.parse_args(argv)

    samples = scan_mc_samples(Path(args.mc_base), year=args.year, channel=args.channel)
    samples = add_data(samples, Path(args.data_dir))

    out_path = PROJECT_ROOT / "Configurations" / args.config_name / "files.json"
    write_files_json(samples, out_path)

    print(f"Saved files.json → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
