from pathlib import Path
import time
import pyarrow.parquet as pq
import pandas as pd

class DataAccess:
    def __init__(self, project_root, sample_config, log_every_files=200):
        self.project_root = project_root
        self.sample_config = sample_config
        self._parquet_read_count = 0
        self.log_every_files = log_every_files

    def _read_parquet(self, file_path, columns=None):
        t0 = time.perf_counter()
        df = pd.read_parquet(file_path, columns=columns)
        dt = time.perf_counter() - t0
        self._parquet_read_count += 1
        if self._parquet_read_count % self.log_every_files == 0:
            print(f"[I/O] read {self._parquet_read_count} files | last={dt:.3f}s | {Path(file_path).name}")
        return df

    def load_parquet(self, file_path, columns=None, selector=None):
        df = self._read_parquet(file_path, columns=columns)
        if selector is not None:
            try:
                df = selector(df)
            except Exception as e:
                print(f"[WARN] Selection failed: {file_path} -> {e}")
        return df

    def get_schema(self, path):
        return pq.ParquetFile(path).schema.names

    def build_index(self):
        index = []
        for sample_name, cfg in self.sample_config.items():
            for d in cfg.get("dirs", []):
                path = (self.project_root / d).resolve()
                if not path.exists():
                    print(f"[WARN] Missing path: {path}")
                    continue
                for file in path.rglob("*.parquet"):
                    schema = self.get_schema(file)
                    index.append({
                        "path": file,
                        "name": file.stem,
                        "sample": sample_name,
                        "kind": cfg.get("kind", "mc"),
                        "scale": cfg.get("scale", 1.0),
                        "color": cfg.get("color", None),
                        "schema": set(schema),
                    })
        return index