from pathlib import Path
import logging
import time

import pyarrow.parquet as pq
import pandas as pd


class DataAccess:
    def __init__(self, project_root, sample_config, log_every_files=200):
        self.project_root = project_root
        self.sample_config = sample_config
        self._table_read_count = 0
        self._schema_read_count = 0
        self.log_every_files = log_every_files

        # Indexing can look 'stuck' on EOS while reading parquet metadata.
        # Keep this coarse to avoid spamming logs.
        self.log_every_schema = 500
        self.slow_schema_seconds = 2.0

        self.logger = logging.getLogger(__name__)

    def _read_table(self, file_path, columns=None):
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        t0 = time.perf_counter()
        if suffix == ".parquet":
            df = pd.read_parquet(file_path, columns=columns)
        elif suffix == ".csv":
            df = pd.read_csv(file_path, usecols=columns)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        dt = time.perf_counter() - t0
        self._table_read_count += 1

        if self.log_every_files and (
            self._table_read_count == 1 or self._table_read_count % self.log_every_files == 0
        ):
            col_msg = "all" if columns is None else str(len(columns))
            self.logger.info(
                "I/O read %d files | last=%.3fs | cols=%s | %s",
                self._table_read_count,
                dt,
                col_msg,
                file_path.name,
            )

        return df

    def load_parquet(self, file_path, columns=None, selector=None):
        """Load a table (parquet/csv) and optionally apply a selector."""
        df = self._read_table(file_path, columns=columns)
        if selector is not None:
            try:
                df = selector(df)
            except Exception as e:
                self.logger.warning("Selection failed: %s -> %s", file_path, e)
        return df

    def get_schema(self, path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pq.ParquetFile(path).schema.names
        if suffix == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
        raise ValueError(f"Unsupported file type: {path}")

    def build_index(self):
        index = []
        for sample_name, cfg in self.sample_config.items():
            kind = cfg.get("kind", "mc")
            scale = cfg.get("scale", 1.0)
            color = cfg.get("color", None)

            # files.json can define either:
            #  - "dirs":  list of directories to scan for *.parquet/*.csv
            #  - "files": explicit list of files
            files_cfg = cfg.get("files")
            if files_cfg is not None:
                for f in files_cfg:
                    p = Path(f)
                    if not p.is_absolute():
                        p = self.project_root / p
                    file = p.resolve()

                    if not file.exists():
                        self.logger.warning("Missing file: %s", file)
                        continue

                    self._schema_read_count += 1
                    if self.log_every_schema and (self._schema_read_count % self.log_every_schema == 0):
                        self.logger.info(
                            "Schema reads: %d | last=%s",
                            self._schema_read_count,
                            file.name,
                        )

                    t0 = time.perf_counter()
                    schema = self.get_schema(file)
                    dt = time.perf_counter() - t0
                    if self.slow_schema_seconds and dt >= self.slow_schema_seconds:
                        self.logger.warning("Slow schema read: %.2fs | %s", dt, file)

                    index.append(
                        {
                            "path": file,
                            "name": file.stem,
                            "sample": sample_name,
                            "kind": kind,
                            "scale": scale,
                            "color": color,
                            "schema": set(schema),
                        }
                    )

                continue

            for d in cfg.get("dirs", []):
                path = (self.project_root / d).resolve()
                if not path.exists():
                    self.logger.warning("Missing path: %s", path)
                    continue

                for file in path.rglob("*"):
                    if not file.is_file():
                        continue
                    if file.suffix.lower() not in {".parquet", ".csv"}:
                        continue

                    self._schema_read_count += 1
                    if self.log_every_schema and (self._schema_read_count % self.log_every_schema == 0):
                        self.logger.info(
                            "Schema reads: %d | last=%s",
                            self._schema_read_count,
                            file.name,
                        )

                    t0 = time.perf_counter()
                    schema = self.get_schema(file)
                    dt = time.perf_counter() - t0
                    if self.slow_schema_seconds and dt >= self.slow_schema_seconds:
                        self.logger.warning("Slow schema read: %.2fs | %s", dt, file)

                    index.append(
                        {
                            "path": file,
                            "name": file.stem,
                            "sample": sample_name,
                            "kind": kind,
                            "scale": scale,
                            "color": color,
                            "schema": set(schema),
                        }
                    )

        return index
