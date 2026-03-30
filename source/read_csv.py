from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_events_csv(csv_data):

    df = pd.read_csv(csv_data)

    event_df = df[['H.m', 'H.pt', 'METx', 'METy', 'covXX', 'covXY', 'covYY', 'dm1', 'pt1', 'eta1', 'phi1', 'mass1', 'type1', 'dm2', 'pt2', 'eta2', 'phi2', 'mass2', 'type2']].copy()

    Higgs_mass = event_df.pop('H.m').to_numpy()
    Higgs_pt = event_df.pop('H.pt').to_numpy()
    METx = event_df.pop('METx').to_numpy()
    METy = event_df.pop('METy').to_numpy()
    metcov = event_df[['covXX', 'covXY', 'covXY', 'covYY']].to_numpy()
    event_df.drop(columns=['covXX', 'covXY', 'covYY'], inplace=True)
    metcov = np.reshape(metcov, (len(metcov), 2, 2))

    print(f"selected event_df shape: {event_df.shape}")
    print("event_df head:\n", event_df.head())

    events = event_df.to_numpy()
    events = np.reshape(events, (len(events), 2, 6))

    return {"measuredTauLeptons": events, "measuredMETx": METx, "measuredMETy": METy, "covMET": metcov, "Higgs_mass": Higgs_mass, "Higgs_pt": Higgs_pt}

def load_input_file(file_path, tree_name=None, branches=None):
    if file_path.endswith(".root"):
        if tree_name is None or branches is None:
            raise ValueError("ROOT file should have tree_name and branches")
        return load_root_events(file_path, tree_name, branches)

    elif file_path.endswith(".csv"):
        return load_events_csv(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quick sanity-check for reading input events")
    parser.add_argument(
        "--file",
        default=None,
        help="Path to input file (default: data/Higgs.csv relative to repo root)",
    )
    args = parser.parse_args(argv)

    default_csv = Path(__file__).resolve().parents[1] / "data" / "Higgs.csv"
    file_path = Path(args.file) if args.file is not None else default_csv

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}. "
            f"Pass --file /path/to/file.csv or place it at {default_csv}"
        )

    print(f"Reading CSV with pandas: {file_path}")
    df = pd.read_csv(file_path)
    print(f"pandas shape: {df.shape}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")

    payload = load_input_file(str(file_path))

    events = payload["measuredTauLeptons"]
    metx = payload["measuredMETx"]
    mety = payload["measuredMETy"]
    cov = payload["covMET"]
    hm = payload["Higgs_mass"]
    hpt = payload["Higgs_pt"]

    print("\nConverted to numpy:")
    print(f"measuredTauLeptons shape: {events.shape} (len={len(events)})")
    print(f"measuredMETx shape: {metx.shape} (len={len(metx)})")
    print(f"measuredMETy shape: {mety.shape} (len={len(mety)})")
    print(f"covMET shape: {cov.shape} (len={len(cov)})")
    print(f"Higgs_mass shape: {hm.shape} (len={len(hm)})")
    print(f"Higgs_pt shape: {hpt.shape} (len={len(hpt)})")

    n = len(events)
    ok = (
        events.shape == (n, 2, 6)
        and metx.shape == (n,)
        and mety.shape == (n,)
        and cov.shape == (n, 2, 2)
        and hm.shape == (n,)
        and hpt.shape == (n,)
    )
    print(f"\nBasic shape check: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())