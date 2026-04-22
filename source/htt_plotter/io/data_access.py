from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import logging
import time

import pandas as pd

from rich.console import Console
from rich.live import Live
from rich.table import Table


class DataAccess:
    """I/O layer.

    Performance notes:
    - Prefer explicit file lists ("files": [...]) in files.json.
    - Schema is computed once per sample (not per file).
    - Parquet reading supports pyarrow.dataset batch scanning with threads.
    """

    def __init__(
        self,
        project_root,
        sample_config,
        log_every_files=200,
        *,
        max_cached_datasets: int = 1,
        max_files_per_dataset: int | None = 1024,
        schema_cache_dir: Path | None = None,
    ):
        self.project_root = Path(project_root)
        self.sample_config = sample_config
        self.log_every_files = log_every_files
        self.max_cached_datasets = max_cached_datasets
        self.max_files_per_dataset = max_files_per_dataset

        # Cache schema locally to avoid repeatedly opening parquet footers on EOS.
        self.schema_cache_dir = (
            Path(schema_cache_dir)
            if schema_cache_dir is not None
            else (self.project_root / ".cache" / "htt_plotter" / "schema")
        )

        self._table_read_count = 0
        self._schema_read_count = 0

        self.logger = logging.getLogger(__name__)

        # Cache can retain fragment metadata for many files; keep it bounded.
        self._dataset_cache: OrderedDict[str, object] = OrderedDict()

    def _schema_cache_path(self, sample: str) -> Path:
        safe = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in sample)
        return self.schema_cache_dir / f"{safe}.json"

    @staticmethod
    def _file_sig(p: Path) -> dict:
        st = os.stat(p)
        return {"path": str(p), "mtime_ns": st.st_mtime_ns, "size": st.st_size}

    def _try_load_cached_schema(self, *, sample: str, files: list[Path]) -> list[str] | None:
        try:
            self.schema_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        cache_path = self._schema_cache_path(sample)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        cached_sig = payload.get("sig")
        cached_cols = payload.get("columns")
        cached_nfiles = payload.get("nfiles")
        if not isinstance(cached_sig, dict) or not isinstance(cached_cols, list) or not isinstance(cached_nfiles, int):
            return None

        try:
            sig = self._file_sig(files[0])
        except Exception:
            return None

        if cached_nfiles != len(files):
            return None
        if cached_sig != sig:
            return None
        if not all(isinstance(c, str) for c in cached_cols):
            return None

        return cached_cols

    def _try_store_cached_schema(self, *, sample: str, files: list[Path], columns: list[str]) -> None:
        try:
            self.schema_cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "sig": self._file_sig(files[0]),
                "nfiles": len(files),
                "columns": list(columns),
            }
            self._schema_cache_path(sample).write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            return

    def _resolve_files(self, files_cfg) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        for f in files_cfg or []:
            p = Path(f)
            if not p.is_absolute():
                p = self.project_root / p
            p = p.absolute()
            key = p.as_posix()
            if key in seen:
                continue
            seen.add(key)
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

    def _sample_schema(self, fmt: str, files: list[Path], *, sample: str | None = None) -> list[str]:
        t0 = time.perf_counter()
        if fmt == "parquet":
            # Avoid constructing a dataset over thousands of fragments just to
            # read schema: it can be slow and memory-hungry.
            if sample is not None:
                cached = self._try_load_cached_schema(sample=sample, files=files)
                if cached is not None:
                    schema = cached
                else:
                    import pyarrow.parquet as pq

                    schema = list(pq.read_schema(files[0]).names)
                    self._try_store_cached_schema(sample=sample, files=files, columns=schema)
            else:
                import pyarrow.parquet as pq

                schema = list(pq.read_schema(files[0]).names)
        elif fmt == "csv":
            schema = list(pd.read_csv(files[0], nrows=0).columns)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        dt = time.perf_counter() - t0
        self._schema_read_count += 1
        if fmt == "parquet":
            self.logger.debug("Schema read (parquet): %.3fs | %d cols", dt, len(schema))
        else:
            self.logger.debug("Schema read (%s): %.3fs | %d cols", fmt, dt, len(schema))
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

            # Common production pattern: a directory contains job*.parquet and a merged.parquet
            # which is the concatenation of the job shards. If both are listed, selecting both
            # will double-count. Prefer merged.parquet to reduce file opens.
            merged = [p for p in files if p.name == "merged.parquet"]
            if merged and len(files) > len(merged):
                self.logger.warning(
                    "Sample '%s' lists merged.parquet together with %d other files; using merged.parquet only to avoid double-counting.",
                    sample_name,
                    len(files) - len(merged),
                )
                files = merged

            if not files:
                self.logger.warning("No input files for sample: %s", sample_name)
                continue

            fmt = self._infer_format(files)

            console = Console()
            history = []

            table = Table(title="Indexing samples")
            table.add_column("Sample")
            table.add_column("Kind")
            table.add_column("Source")
            table.add_column("Format")
            table.add_column("Files")

            index = []

            sample_config = self.sample_config.get("samples", self.sample_config)

            with Live(table, console=console, refresh_per_second=4) as live:

                for sample_name, cfg in sample_config.items():

                    kind = cfg.get("kind", "mc")
                    scale = cfg.get("scale", 1.0)
                    color = cfg.get("color", None)

                    files_cfg = cfg.get("files")
                    dirs_cfg = cfg.get("dirs")

                    source_kind = "files" if files_cfg else "dirs"

                    if files_cfg is not None:
                        files = self._resolve_files(files_cfg)
                    else:
                        if dirs_cfg:
                            self.logger.warning(
                                "Sample '%s' uses 'dirs' scanning; generate explicit 'files' list for speed.",
                                sample_name,
                            )
                        files = self._scan_dirs(dirs_cfg)

                    files = [p for p in files if p.exists()]

                    merged = [p for p in files if p.name == "merged.parquet"]
                    if merged and len(files) > len(merged):
                        self.logger.warning(
                            "Sample '%s' uses merged.parquet only to avoid double counting.",
                            sample_name,
                        )
                        files = merged

                    if not files:
                        self.logger.warning("No input files for sample: %s", sample_name)
                        continue

                    fmt = self._infer_format(files)

                    history.append((sample_name, kind, source_kind, fmt, len(files)))

                    table = Table(title="Indexing samples")
                    table.add_column("Sample")
                    table.add_column("Kind")
                    table.add_column("Source")
                    table.add_column("Format")
                    table.add_column("Files")

                    for row in history:
                        table.add_row(*map(str, row))

                    live.update(table)

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
            def _file_chunks(all_files: list[Path]) -> list[list[Path]]:
                if self.max_files_per_dataset is None or self.max_files_per_dataset <= 0:
                    return [all_files]
                if len(all_files) <= self.max_files_per_dataset:
                    return [all_files]
                return [
                    all_files[i : i + self.max_files_per_dataset]
                    for i in range(0, len(all_files), self.max_files_per_dataset)
                ]

            file_chunks = _file_chunks(files)
            if len(file_chunks) > 1:
                self.logger.info(
                    "Parquet scan chunking: sample=%s | files=%d | chunks=%d | max_files_per_dataset=%d",
                    cache_key,
                    len(files),
                    len(file_chunks),
                    int(self.max_files_per_dataset),
                )

            dataset = None
            use_cache = (
                self.max_cached_datasets is not None
                and self.max_cached_datasets > 0
                and len(file_chunks) == 1
            )
            if use_cache:
                dataset = self._dataset_cache.get(cache_key)
                if dataset is not None:
                    self._dataset_cache.move_to_end(cache_key)

            col_msg = "all" if columns is None else str(len(columns))
            self.logger.info(
                "Parquet scan start: sample=%s | files=%d | cols=%s | filter=%s | batch_size=%d",
                cache_key,
                len(files),
                col_msg,
                "yes" if filter_expr is not None else "no",
                batch_size,
            )

            t0 = time.perf_counter()
            last_log = t0
            batches = 0
            rows = 0

            # Timing breakdown:
            # - io_time_s: time spent waiting for Arrow to produce the next batch
            # - consumer_time_s: time spent in caller between yields (hist filling, etc.)
            io_time_s = 0.0
            consumer_time_s = 0.0
            resume_time = t0

            for chunk_idx, chunk_files in enumerate(file_chunks, start=1):
                if dataset is None or len(file_chunks) > 1:
                    dataset = ds.dataset([str(p) for p in chunk_files], format="parquet")
                    if use_cache:
                        self._dataset_cache[cache_key] = dataset
                        self._dataset_cache.move_to_end(cache_key)
                        while len(self._dataset_cache) > self.max_cached_datasets:
                            self._dataset_cache.popitem(last=False)

                scanner = dataset.scanner(
                    columns=columns,
                    filter=filter_expr,
                    use_threads=True,
                    batch_size=batch_size,
                )

                if len(file_chunks) > 1:
                    self.logger.info(
                        "Parquet scan chunk start: sample=%s | chunk=%d/%d | files=%d",
                        cache_key,
                        chunk_idx,
                        len(file_chunks),
                        len(chunk_files),
                    )

                for batch in scanner.to_batches():
                    batch_ready = time.perf_counter()
                    io_time_s += batch_ready - resume_time

                    batches += 1
                    rows += batch.num_rows

                    now = batch_ready
                    if (now - last_log) >= 10.0:
                        self.logger.info(
                            "Parquet scan progress: sample=%s | batches=%d | rows=%d | elapsed=%.1fs | io=%.1fs | consumer=%.1fs",
                            cache_key,
                            batches,
                            rows,
                            now - t0,
                            io_time_s,
                            consumer_time_s,
                        )
                        last_log = now

                    yield batch

                    resume_time = time.perf_counter()
                    consumer_time_s += resume_time - batch_ready

            total_time_s = time.perf_counter() - t0
            self.logger.info(
                "Parquet scan done: sample=%s | batches=%d | rows=%d | elapsed=%.1fs | io=%.1fs | consumer=%.1fs",
                cache_key,
                batches,
                rows,
                total_time_s,
                io_time_s,
                consumer_time_s,
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
