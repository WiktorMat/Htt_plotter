from __future__ import annotations

from pathlib import Path
import logging
import time

import pandas as pd


class DataAccess:
    """I/O layer.

    Performance notes:
    - Prefer explicit file lists ("files": [...]) in files.json.
    - Schema is computed once per sample (not per file).
    - Parquet reading supports pyarrow.dataset batch scanning with threads.
    """

    def __init__(self, project_root, sample_config, log_every_files=200):
        self.project_root = Path(project_root)
        self.sample_config = sample_config
        self.log_every_files = log_every_files

        self._table_read_count = 0
        self._schema_read_count = 0

        self.logger = logging.getLogger(__name__)

        self._dataset_cache: dict[str, object] = {}

    def _resolve_files(self, files_cfg) -> list[Path]:
        files: list[Path] = []
        for f in files_cfg or []:
            p = Path(f)
            if not p.is_absolute():
                p = self.project_root / p
            p = p.absolute()
            files.append(p)
        return files

    def _scan_dirs(self, dirs_cfg) -> list[Path]:
        files: list[Path] = []
        for d in dirs_cfg or []:
            path = (self.project_root / d).resolve()
            if not path.exists():
                self.logger.warning("Missing path: %s", path)
                continue

            for file in path.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in {".parquet", ".csv"}:
                    continue
                files.append(file.resolve())

        return files

    def _infer_format(self, files: list[Path]) -> str:
        if not files:
            raise ValueError("No files")
        suffixes = {p.suffix.lower() for p in files}
        if len(suffixes) != 1:
            raise ValueError(f"Mixed file types not supported: {sorted(suffixes)}")
        s = next(iter(suffixes))
        if s == ".parquet":
            return "parquet"
        if s == ".csv":
            return "csv"
        raise ValueError(f"Unsupported file type: {s}")

    def _sample_schema(self, fmt: str, files: list[Path]) -> list[str]:
        t0 = time.perf_counter()
        if fmt == "parquet":
            import pyarrow.dataset as ds

            dataset = ds.dataset([str(p) for p in files], format="parquet")
            schema = list(dataset.schema.names)
        elif fmt == "csv":
            schema = list(pd.read_csv(files[0], nrows=0).columns)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        dt = time.perf_counter() - t0
        self._schema_read_count += 1
        self.logger.info("Schema read (%s): %.3fs | %d cols", fmt, dt, len(schema))
        return schema

    def build_index(self) -> list[dict]:
        """Return per-sample index entries.

        Each entry:
            sample, kind, scale, color, files, format, schema
        """

        index: list[dict] = []

        sample_config = self.sample_config.get("samples", self.sample_config)
        process_config = self.sample_config.get("process", {})

        for sample_name, cfg in sample_config.items():
            kind = cfg.get("kind", "mc")
            scale = cfg.get("scale", 1.0)
            color = cfg.get("color", None)

            files_cfg = cfg.get("files")
            dirs_cfg = cfg.get("dirs")

            if files_cfg is not None:
                source_kind = "files"
                files = self._resolve_files(files_cfg)
            else:
                source_kind = "dirs"
                # Backward compatible, but slow on large EOS trees.
                if dirs_cfg:
                    self.logger.warning(
                        "Sample '%s' uses 'dirs' scanning; generate explicit 'files' list for speed.",
                        sample_name,
                    )
                files = self._scan_dirs(dirs_cfg)

            files = [p for p in files if p.exists()]
            if not files:
                self.logger.warning("No input files for sample: %s", sample_name)
                continue

            fmt = self._infer_format(files)
            self.logger.info(
                "Index sample '%s': kind=%s | source=%s | format=%s | files=%d",
                sample_name,
                kind,
                source_kind,
                fmt,
                len(files),
            )
            schema = self._sample_schema(fmt, files)

            index.append(
                {
                    "sample": sample_name,
                    "kind": kind,
                    "scale": scale,
                    "color": color,
                    "files": files,
                    "format": fmt,
                    "schema": set(schema),
                }
            )

        return index

    def iter_batches(
        self,
        index_item: dict,
        *,
        columns: list[str] | None = None,
        filter_expr=None,
        batch_size: int = 131072,
    ):
        """Iterate pyarrow RecordBatches for a sample.

        For parquet this uses pyarrow.dataset with threaded scanning.
        For csv it falls back to pandas per-file (still yields batches).
        """

        fmt = index_item["format"]
        files: list[Path] = index_item["files"]

        if fmt == "parquet":
            import pyarrow.dataset as ds

            cache_key = index_item["sample"]
            dataset = self._dataset_cache.get(cache_key)
            if dataset is None:
                dataset = ds.dataset([str(p) for p in files], format="parquet")
                self._dataset_cache[cache_key] = dataset

            col_msg = "all" if columns is None else str(len(columns))
            self.logger.info(
                "Parquet scan start: sample=%s | files=%d | cols=%s | filter=%s | batch_size=%d",
                cache_key,
                len(files),
                col_msg,
                "yes" if filter_expr is not None else "no",
                batch_size,
            )

            scanner = dataset.scanner(
                columns=columns,
                filter=filter_expr,
                use_threads=True,
                batch_size=batch_size,
            )

            t0 = time.perf_counter()
            last_log = t0
            batches = 0
            rows = 0

            for batch in scanner.to_batches():
                batches += 1
                rows += batch.num_rows

                now = time.perf_counter()
                if (now - last_log) >= 10.0:
                    self.logger.info(
                        "Parquet scan progress: sample=%s | batches=%d | rows=%d | elapsed=%.1fs",
                        cache_key,
                        batches,
                        rows,
                        now - t0,
                    )
                    last_log = now

                yield batch

            self.logger.info(
                "Parquet scan done: sample=%s | batches=%d | rows=%d | elapsed=%.1fs",
                cache_key,
                batches,
                rows,
                time.perf_counter() - t0,
            )

            return

        if fmt == "csv":
            import pyarrow as pa

            sample = index_item.get("sample", "<unknown>")
            col_msg = "all" if columns is None else str(len(columns))
            self.logger.info(
                "CSV scan start: sample=%s | files=%d | cols=%s",
                sample,
                len(files),
                col_msg,
            )

            for p in files:
                t0 = time.perf_counter()
                df = pd.read_csv(p, usecols=columns)
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
                        p.name,
                    )

                table = pa.Table.from_pandas(df, preserve_index=False)
                for batch in table.to_batches(max_chunksize=batch_size):
                    yield batch

            self.logger.info(
                "CSV scan done: sample=%s | files=%d",
                sample,
                len(files),
            )

            return

        raise ValueError(f"Unsupported format: {fmt}")
