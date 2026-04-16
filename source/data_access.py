import pyarrow.dataset as ds
from functools import reduce
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import logging
from pathlib import Path
import time
from Configurations.config_0.Config import *
from Selection import SELECT

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

    def _convert_selector_to_arrow(self):
        exprs = []

        if PT_1_CUT is not None:
            exprs.append(ds.field("pt_1") > PT_1_CUT)

        if PT_2_CUT is not None:
            exprs.append(ds.field("pt_2") > PT_2_CUT)

        if ETA_1_CUT is not None:
            exprs.append(pc.abs(ds.field("eta_1")) < ETA_1_CUT)

        if ETA_2_CUT is not None:
            exprs.append(pc.abs(ds.field("eta_2")) < ETA_2_CUT)

        if not exprs:
            return None

        return reduce(lambda a, b: pc.and_(a, b), exprs)
    
    def _read_parquet(self, file_path, columns=None):
        t0 = time.perf_counter()
        df = pd.read_parquet(file_path, columns=columns)
        dt = time.perf_counter() - t0
        self._parquet_read_count += 1
        if self._parquet_read_count % self.log_every_files == 0:
            print(f"I/O read {self._parquet_read_count} files | last={dt:.3f}s | {Path(file_path).name}")
        return df

    def load_dataset(self, path, columns=None, selector=None):
        ext = Path(path).suffix.lower()

        if ext == ".parquet":
            table = ds.dataset(path, format="parquet").to_table(columns=columns)
            df = table.to_pandas()

            if selector is not None:
                df = selector(df)

            return df

        elif ext == ".csv":
            return self.load_csv(path, columns=columns, selector=selector)

        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def load_csv(self, path, columns=None, selector=None):
        df = pd.read_csv(path)

        if columns is not None:
            columns = [c for c in columns if c in df.columns]
            df = df[columns]

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

                sample = file_path.parent.name

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