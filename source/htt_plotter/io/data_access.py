from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import logging
import time
from typing import Any, Callable

import pandas as pd

from htt_plotter.io.file_discovery import resolve_files, scan_dirs
from htt_plotter.io.indexing import build_index as build_index_impl
from htt_plotter.io.prefetch import iter_batches_from_items
from htt_plotter.io.schema_cache import try_load_cached_schema, try_store_cached_schema


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

    def _resolve_files(self, files_cfg) -> list[Path]:
        return resolve_files(self.project_root, files_cfg)

    def _scan_dirs(self, dirs_cfg) -> list[Path]:
        return scan_dirs(self.project_root, dirs_cfg, logger=self.logger)

    def _infer_format(self, files: list[Path]) -> str:
        if not files:
            files = [Path(f) for f in files if f]
            if not files:
                self.logger.warning("Empty file list after cleaning → fallback to 'unknown'")
                return "unknown"
            if len(suffixes) != 1:
                self.logger.warning("Mixed file types → fallback to first file type")
                return next(iter(suffixes))
        suffixes = {Path(p).suffix.lower() for p in files if p}
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
                cached = try_load_cached_schema(
                    schema_cache_dir=self.schema_cache_dir,
                    sample=sample,
                    files=files,
                )
                if cached is not None:
                    schema = cached
                else:
                    import pyarrow.parquet as pq

                    schema = list(pq.read_schema(files[0]).names)
                    if not files:
                        self.logger.warning("No files for schema → returning empty schema")
                        return []
                    
                    try:
                        schema = list(pq.read_schema(files[0]).names)
                    except Exception as e:
                        self.logger.warning("Schema read failed → %s", e)
                        return []
                    
                    try_store_cached_schema(
                        schema_cache_dir=self.schema_cache_dir,
                        sample=sample,
                        files=files,
                        columns=schema,
                    )
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
        return build_index_impl(
            project_root=self.project_root,
            sample_config=self.sample_config,
            logger=self.logger,
            resolve_files=self._resolve_files,
            scan_dirs=self._scan_dirs,
            infer_format=self._infer_format,
            sample_schema=self._sample_schema,
        )

    def iter_batches(
        self,
        index_item: dict,
        *,
        columns: list[str] | None = None,
        filter_expr=None,
        batch_size: int = 131072,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_s: float = 10.0,
        prefetch_batches: int = 0,
    ):
        """Iterate pyarrow RecordBatches for a sample.

        For parquet this uses pyarrow.dataset with threaded scanning.
        For csv it falls back to pandas per-file (still yields batches).

        Parallelization option:
        - If prefetch_batches > 0, the underlying scan runs in a background
          thread and batches are buffered in a bounded queue. This overlaps
          I/O with downstream processing while keeping callbacks executed in
          the consumer (main) thread.
        """

        fmt = index_item.get("format")
        files = index_item.get("files", [])

        if not fmt:
            raise ValueError(f"Missing format in index_item: {index_item}")

        if not files:
            self.logger.warning("Empty file list for sample=%s → skipping", index_item.get("sample"))
            return

        if fmt == "parquet":
            import pyarrow.dataset as ds

            cache_key = index_item["sample"]

            def _file_chunks(all_files: list[Path]) -> list[list[Path]]:
                all_files = [p for p in all_files if p]
                if not all_files:
                    return [[]]
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
                self.logger.debug(
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
                if dataset is not None and not hasattr(dataset, "scanner"):
                    self.logger.warning("Corrupted cache entry for %s → dropping", cache_key)
                    dataset = None
                    self._dataset_cache.pop(cache_key, None)
                if dataset is not None:
                    self._dataset_cache.move_to_end(cache_key)

            def _parquet_items():
                nonlocal dataset
                col_msg = "all" if columns is None else str(len(columns))
                yield (
                    "event",
                    {
                        "event": "start",
                        "sample": cache_key,
                        "files": len(files),
                        "cols": col_msg,
                        "filter": "yes" if filter_expr is not None else "no",
                        "batch_size": batch_size,
                        "chunks": len(file_chunks),
                    },
                )

                t0 = time.perf_counter()
                last_log = t0
                batches = 0
                rows = 0

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

                    try:
                        scanner = dataset.scanner(
                            columns=columns,
                            filter=filter_expr,
                            use_threads=True,
                            batch_size=batch_size,
                        )
                    except Exception as e:
                        failed_files = 0
                        failed_files += 1
                        self.logger.error("Scanner failed for %s → %s", cache_key, e)
                        return
                
                if failed_files == len(files):
                    self.logger.error("All CSV files failed → returning empty dataset")
                    return

                    if len(file_chunks) > 1:
                        yield (
                            "event",
                            {
                                "event": "chunk_start",
                                "sample": cache_key,
                                "chunk": chunk_idx,
                                "chunks": len(file_chunks),
                                "chunk_files": len(chunk_files),
                            },
                        )

                    for batch in scanner.to_batches():
                        if batch is None:
                            self.logger.warning("Null batch in %s → skipping", cache_key)
                            continue
                        batch_ready = time.perf_counter()
                        io_time_s += batch_ready - resume_time

                        batches += 1
                        rows += batch.num_rows

                        now = batch_ready
                        if (now - last_log) >= float(progress_interval_s):
                            payload: dict[str, Any] = {
                                "event": "progress",
                                "sample": cache_key,
                                "batches": batches,
                                "rows": rows,
                                "elapsed_s": now - t0,
                                "io_s": io_time_s,
                                "consumer_s": consumer_time_s,
                            }
                            if len(file_chunks) > 1:
                                payload["chunk"] = chunk_idx
                                payload["chunks"] = len(file_chunks)
                            yield ("event", payload)
                            last_log = now

                        yield ("batch", batch)

                        # Note: with prefetch enabled, this consumer time measures
                        # how long it took to enqueue the batch, not the caller's
                        # processing time. It's still useful as a rough indicator.
                        resume_time = time.perf_counter()
                        consumer_time_s += resume_time - batch_ready

                total_time_s = time.perf_counter() - t0
                yield (
                    "event",
                    {
                        "event": "done",
                        "sample": cache_key,
                        "batches": batches,
                        "rows": rows,
                        "elapsed_s": total_time_s,
                        "io_s": io_time_s,
                        "consumer_s": consumer_time_s,
                    },
                )

            # In non-prefetch mode, keep the previous debug logging behavior
            # when no progress_callback is provided.
            if progress_callback is None:
                col_msg = "all" if columns is None else str(len(columns))
                self.logger.debug(
                    "Parquet scan start: sample=%s | files=%d | cols=%s | filter=%s | batch_size=%d | prefetch=%d",
                    cache_key,
                    len(files),
                    col_msg,
                    "yes" if filter_expr is not None else "no",
                    batch_size,
                    int(prefetch_batches),
                )

            items = _parquet_items()
            yield from iter_batches_from_items(
                items,
                progress_callback=progress_callback,
                prefetch_batches=prefetch_batches,
            )

            if progress_callback is None:
                self.logger.debug("Parquet scan done: sample=%s", cache_key)

            return

        if fmt == "csv":
            import pyarrow as pa

            sample = index_item.get("sample", "<unknown>")

            def _csv_items():
                col_msg = "all" if columns is None else str(len(columns))
                yield (
                    "event",
                    {
                        "event": "start",
                        "sample": sample,
                        "files": len(files),
                        "cols": col_msg,
                        "batch_size": batch_size,
                    },
                )

                t0 = time.perf_counter()
                batches = 0
                rows = 0
                last_log = t0

                for p in files:
                    read_t0 = time.perf_counter()
                    try:
                        df = pd.read_csv(p, usecols=columns)
                    except Exception as e:
                        self.logger.warning("CSV read failed %s → %s", p, e)
                        continue
                    dt = time.perf_counter() - read_t0
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
                        batches += 1
                        rows += batch.num_rows
                        now = time.perf_counter()
                        if (now - last_log) >= float(progress_interval_s):
                            yield (
                                "event",
                                {
                                    "event": "progress",
                                    "sample": sample,
                                    "batches": batches,
                                    "rows": rows,
                                    "elapsed_s": now - t0,
                                },
                            )
                            last_log = now
                        yield ("batch", batch)

                yield (
                    "event",
                    {
                        "event": "done",
                        "sample": sample,
                        "batches": batches,
                        "rows": rows,
                        "elapsed_s": time.perf_counter() - t0,
                    },
                )

            if progress_callback is None:
                col_msg = "all" if columns is None else str(len(columns))
                self.logger.debug(
                    "CSV scan start: sample=%s | files=%d | cols=%s | prefetch=%d",
                    sample,
                    len(files),
                    col_msg,
                    int(prefetch_batches),
                )

            items = _csv_items()
            yield from iter_batches_from_items(
                items,
                progress_callback=progress_callback,
                prefetch_batches=prefetch_batches,
            )

            if progress_callback is None:
                self.logger.debug("CSV scan done: sample=%s | files=%d", sample, len(files))

            return

        raise ValueError(f"Unsupported format: {fmt}")
