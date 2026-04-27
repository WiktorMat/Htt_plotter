from pathlib import Path
import json
import pyarrow.csv as pv
import pyarrow.parquet as pq

project_root = Path(__file__).resolve().parents[3]

files_json = project_root / "Configurations" / "config_0" / "files.json"

def safe_load_json(path: Path, default):
    if not path.exists():
        print(f"[WARN] Missing config: {path} → using empty data")
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")
        return default

def read_schema(file_path: str):
    path = Path(file_path)

    try:
        if path.suffix == ".parquet":
            return pq.read_table(path).schema.names

        if path.suffix == ".csv":
            return pv.read_csv(path).schema.names

        raise ValueError(f"Unsupported file type: {path.suffix}")

    except Exception as e:
        print(f"[WARN] Failed reading schema {path}: {e}")
        return []


def main():
    data = safe_load_json(files_json, default={})

    mc_sample = None
    data_sample = None

    if not isinstance(data, dict):
        print("[ERROR] files.json is not a valid dict → aborting")
        return

    for name, sample in data.items():
        kind = sample.get("kind", "mc")

        if kind == "mc" and mc_sample is None:
            mc_sample = (name, sample)

        if kind == "data" and data_sample is None:
            data_sample = (name, sample)

    def process(entry, label):
        if not entry:
            print(f"\n[{label}] not found\n")
            return

        name, sample = entry

        dirs = sample.get("dirs") or []
        if not isinstance(dirs, list):
            print(f"[WARN] Invalid dirs format in {name} → expected list")
            dirs = []

        folder = project_root / dirs[0]

        print(f"\n===== {label}: {name} =====")
        print(f"Folder: {folder}")

        if not folder.exists() or not folder.is_dir():
            print(f"[WARN] Folder invalid or missing: {folder}")
            return

        parquet_files = list(folder.glob("*.parquet"))
        csv_files = list(folder.glob("*.csv"))

        files = parquet_files + csv_files

        if parquet_files:
            file_path = parquet_files[0]
        elif csv_files:
            file_path = csv_files[0]
        else:
            print("No data files found in folder")
            return

        if not files:
            print("No data files found in folder")
            return

        file_path = files[0]

        print(f"Using file: {file_path}")

        cols = read_schema(file_path)

        print("\nColumns:")

        if not cols:
            print("No columns detected (empty or unreadable file)")
            return
        for c in cols:
            print(" -", c)

        print("\nTotal:", len(cols))

    process(mc_sample, "MC")
    process(data_sample, "DATA")


if __name__ == "__main__":
    main()