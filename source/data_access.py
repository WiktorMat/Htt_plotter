from pathlib import Path
import time
import pyarrow.parquet as pq
import pandas as pd
import logging
from Configurations.config_0.Config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
            logging.StreamHandler(),
            logging.FileHandler("plotter.log")
        ]
)

class DataAccess:
    def __init__(self, project_root, sample_config, log_every_files=200):
        self.project_root = project_root
        self.sample_config = sample_config
        self._parquet_read_count = 0
        self.log_every_files = log_every_files
        self.base_path = (self.project_root / "data").resolve()

    def _read_parquet(self, file_path, columns=None):
        t0 = time.perf_counter()
        df = pd.read_parquet(file_path, columns=columns)
        dt = time.perf_counter() - t0
        self._parquet_read_count += 1
        if self._parquet_read_count % self.log_every_files == 0:
            print(f"I/O read {self._parquet_read_count} files | last={dt:.3f}s | {Path(file_path).name}")
        return df

    def load_parquet(self, file_path, columns=None, selector=None):
        df = self._read_parquet(file_path, columns=columns)
        if selector is not None:
            try:
                df = selector(df)
            except Exception as e:
                logging.warning(f"Selection failed: {file_path} -> {e}")
        return df
    
    def load_csv(self, path, columns=None, selector=None):
        df = pd.read_csv(path, usecols=columns)

        if selector is not None:
            try:
                df = selector(df)
            except Exception:
                pass

        return df

    def get_schema(self, path):
        return pq.ParquetFile(path).schema.names

    def build_index(self):
        index = []

        logging.info(f"BASE PATH: {self.base_path}")

        if not self.base_path.exists():
            logging.warning("PATH DOES NOT EXIST!")
            return index

        files = list(self.base_path.rglob("*"))

        logging.info(f"ALL FILES FOUND: {len(files)}")

        for file_path in files:
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            logging.info(f"CHECK: {file_path}")

            if ext not in [".csv", ".parquet"]:
                continue

            try:
                if ext == ".csv":
                    schema = set(pd.read_csv(file_path, nrows=0).columns)

                elif ext == ".parquet":
                    schema = set(pd.read_parquet(file_path).columns)

                sample = next((p for p in file_path.parts if p in process), file_path.parent.name)

                index.append({
                    "path": file_path,
                    "sample": sample,
                    "schema": schema,
                    "kind": "data" if "data" in sample.lower() else "mc"
                })

                logging.info(f"ADDED: {file_path}")

            except Exception as e:
                logging.warning(f"ERROR:: {file_path}, {e}")

        return index