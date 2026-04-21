from pathlib import Path
import json
import pyarrow.csv as pv
import pyarrow.parquet as pq

project_root = Path(__file__).resolve().parents[3]

files_json = project_root / "Configurations" / "config_0" / "files.json"


def read_schema(file_path: str):
    path = Path(file_path)

    if path.suffix == ".parquet":
        return pq.read_table(path).schema.names

    if path.suffix == ".csv":
        return pv.read_csv(path).schema.names

    raise ValueError(f"Unsupported file type: {path.suffix}")


def main():
    with open(files_json) as f:
        data = json.load(f)

    mc_sample = None
    data_sample = None

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

        dirs = sample.get("dirs", [])
        if not dirs:
            print(f"\n[{label}] has no dirs\n")
            return

        folder = project_root / dirs[0]

        print(f"\n===== {label}: {name} =====")
        print(f"Folder: {folder}")

        if not folder.exists():
            print("Folder does not exist!")
            return

        files = list(folder.glob("*.parquet")) + list(folder.glob("*.csv"))

        if not files:
            print("No data files found in folder")
            return

        file_path = files[0]

        print(f"Using file: {file_path}")

        cols = read_schema(file_path)

        print("\nColumns:")
        for c in cols:
            print(" -", c)

        print("\nTotal:", len(cols))

    process(mc_sample, "MC")
    process(data_sample, "DATA")


if __name__ == "__main__":
    main()