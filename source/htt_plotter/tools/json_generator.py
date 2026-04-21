from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_ROOT = PROJECT_ROOT / "output"
MC_BASE_DEFAULT = OUTPUT_ROOT / "test_plotter"

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


def _anchor_to_project_root(p: Path) -> Path:
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _format_path(p: Path, *, path_mode: str) -> str:
    if path_mode not in {"relative", "absolute", "auto"}:
        raise ValueError(f"Unsupported path_mode: {path_mode}")

    if path_mode == "absolute":
        return p.as_posix()

    try:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        if path_mode == "relative":
            raise
        return p.as_posix()
    else:
        return rel

def smart_group(name: str) -> str:
    # Target process grouping used by the plotter.
    # Keep this intentionally simple and aligned with config expectations.
    if re.match(r"^Muon\d+$", name):
        return "data"
    if re.match(r"^DYto", name):
        return "DY"
    # tt group includes ttbar and single-top (tW, t-channel) samples
    if re.match(r"^(TT|TW|TbarW|TBbar|TbarB)", name):
        return "tt"
    if re.match(r"^Wto", name):
        return "W+jets"
    return "others"


def build_process_map(samples: dict[str, dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}

    for sample_name in samples.keys():
        group = smart_group(sample_name)

        if group not in groups:
            groups[group] = []

        groups[group].append(sample_name)

    return groups

def add_process_colors(process_map: dict[str, list[str]]) -> dict:
    processes = sorted(process_map.keys())

    return {
        p: {
            "samples": process_map[p],
            "color": get_color(i)
        }
        for i, p in enumerate(processes)
    }

def scan_mc_samples(
    base_dir: Path,
    *,
    year: str = YEAR,
    channel: str = CHANNEL,
    path_mode: str = "auto",
) -> dict:

    base_dir = _anchor_to_project_root(base_dir)
    base_path = base_dir / year / channel

    print(f"Scanning MC base: {base_path}")

    samples: dict[str, dict] = {}

    if not base_path.exists():
        print("MC base does not exist")
        return samples

    for process_dir in sorted([p for p in base_path.iterdir() if p.is_dir()]):

        sample_name = process_dir.name

        files = sorted(process_dir.rglob("*.parquet"))
        if not files:
            continue

        merged = [p for p in files if p.name == "merged.parquet"]
        if merged:
            # Prefer merged parquet to avoid double-counting job shards.
            files = merged

        print(f"  Sample: {sample_name} | files: {len(files)}")

        kind = "data" if smart_group(sample_name) == "data" else "mc"
        samples[sample_name] = {
            "kind": kind,
            "scale": 1.0,
            "files": [_format_path(p, path_mode=path_mode) for p in files],
        }

    print(f"Found samples: {len(samples)}")
    return samples

def write_json(obj: dict, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {name}: {path} | entries={len(obj)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate files.json + process.json")

    parser.add_argument("--config-name", default="config_0")
    parser.add_argument("--year", default=YEAR)
    parser.add_argument("--channel", default=CHANNEL)
    parser.add_argument("--mc-base", default=str(MC_BASE_DEFAULT))

    parser.add_argument(
        "--path-mode",
        choices=["auto", "relative", "absolute"],
        default="auto",
    )

    args = parser.parse_args(argv)

    mc_base = _anchor_to_project_root(Path(args.mc_base))

    samples = scan_mc_samples(
        mc_base,
        year=args.year,
        channel=args.channel,
        path_mode=args.path_mode,
    )

    out_dir = PROJECT_ROOT / "Configurations" / args.config_name
    files_path = out_dir / "files.json"
    write_json(samples, files_path, "files.json")

    process_path = out_dir / "process.json"
    process_map = build_process_map(samples)
    process_json = add_process_colors(process_map)

    write_json(process_json, process_path, "process.json")

    print(f"Saved files.json   → {files_path}")
    print(f"Saved process.json → {process_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())