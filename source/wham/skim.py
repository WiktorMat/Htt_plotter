"""Skim cache: local column-pruned copies of the (badly chunked) inputs.

The source files have thousands of ~1700-row row groups on EOS, so naive
column-pruned reads degenerate into tens of thousands of tiny network
reads. A skim is a one-time rewrite of just the referenced columns with
large row groups; every later run reads the skim sequentially.

Skims carry NO row filter, so editing cuts never invalidates them.
A skim is reusable when the source file signature still matches and the
column set it was built to cover is a superset of what is now required
(columns absent from the source are recorded and never force a rebuild).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from wham import __version__, util
from wham.config import AnalysisConfig, Sample
from wham.parallel import run_parallel
from wham.util import file_signature, short_key

ROW_GROUP_SIZE = 512 * 1024


def coalesced_parquet_format():
    """Parquet format options that merge tiny scattered reads (proven ~2x on EOS)."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    return ds.ParquetFileFormat(
        default_fragment_scan_options=ds.ParquetFragmentScanOptions(
            pre_buffer=True,
            cache_options=pa.CacheOptions(
                hole_size_limit=64 * 1024,
                range_size_limit=32 * 1024 * 1024,
                lazy=False,
            ),
        )
    )


@dataclass(frozen=True)
class SkimInfo:
    sample: str
    path: Path
    columns: frozenset[str]  # columns actually present in the skim
    covers: frozenset[str]   # columns the build was asked for
    src_signature: dict
    rows: int


def skim_dir(analysis_name: str, sample: Sample) -> Path:
    return util.cache_root() / "skims" / analysis_name / sample.name / sample.variation


def _load_manifest(path: Path) -> dict | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or "src" not in manifest:
        return None
    parquet = path.with_suffix(".parquet")
    if not parquet.is_file():
        return None
    return manifest


def _info_from_manifest(sample: str, manifest_path: Path, manifest: dict) -> SkimInfo:
    return SkimInfo(
        sample=sample,
        path=manifest_path.with_suffix(".parquet"),
        columns=frozenset(manifest["columns"]),
        covers=frozenset(manifest["covers"]),
        src_signature=manifest["src"],
        rows=int(manifest.get("rows", -1)),
    )


def find_skim(analysis_name: str, sample: Sample, required: frozenset[str],
              ff_sig: dict | None = None) -> SkimInfo | None:
    """Newest existing skim that matches the source and covers the columns."""
    directory = skim_dir(analysis_name, sample)
    if not directory.is_dir():
        return None
    src_sig = file_signature(sample.path)

    best: tuple[float, Path, dict] | None = None
    for manifest_path in directory.glob("*.json"):
        manifest = _load_manifest(manifest_path)
        if manifest is None or manifest["src"] != src_sig:
            continue
        # appended score columns live in "columns" but not "covers"; either
        # satisfies a requirement
        if not required <= set(manifest["covers"]) | set(manifest["columns"]):
            continue
        if manifest.get("fake_factors") != ff_sig:
            continue
        created = float(manifest.get("created", 0.0))
        if best is None or created > best[0]:
            best = (created, manifest_path, manifest)

    if best is None:
        return None
    return _info_from_manifest(sample.name, best[1], best[2])


def build_skim(analysis_name: str, sample: Sample, required: frozenset[str],
               fake_factors=None, ff_sig: dict | None = None) -> SkimInfo:
    """Read requested columns from the source and rewrite locally (crash-safe)."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    src_sig = file_signature(sample.path)
    available = set(pq.read_schema(sample.path).names)
    columns = sorted(required & available)

    dataset = ds.dataset(str(sample.path), format=coalesced_parquet_format())
    table = dataset.to_table(columns=columns)

    if fake_factors is not None:
        from wham.muffin import augment_table

        table = augment_table(fake_factors, table)

    directory = skim_dir(analysis_name, sample)
    directory.mkdir(parents=True, exist_ok=True)
    key_payload: dict = {"src": src_sig, "covers": sorted(required)}
    if ff_sig is not None:  # conditional so keys without fake factors stay stable
        key_payload["fake_factors"] = ff_sig
    key = short_key(key_payload)
    final = directory / f"{key}.parquet"
    tmp = directory / f"{key}.parquet.tmp.{os.getpid()}"
    pq.write_table(table, tmp, row_group_size=ROW_GROUP_SIZE, compression="snappy")
    os.replace(tmp, final)

    manifest = {
        "src": src_sig,
        "covers": sorted(required),
        "columns": table.column_names,  # includes appended score columns
        "rows": table.num_rows,
        "created": time.time(),
        "version": __version__,
    }
    if ff_sig is not None:
        manifest["fake_factors"] = ff_sig
    (directory / f"{key}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return _info_from_manifest(sample.name, directory / f"{key}.json", manifest)


def ensure_skims(
    cfg: AnalysisConfig,
    samples: list[Sample],
    *,
    workers: int = 6,
    force: bool = False,
    on_progress=None,
) -> dict[str, SkimInfo]:
    """Return a valid skim per sample, building missing ones in parallel."""
    from wham.muffin import signature

    required = cfg.required_columns()
    ff_sig = signature(cfg.fake_factors)

    skims: dict[str, SkimInfo] = {}
    to_build: list[Sample] = []
    for sample in samples:
        info = None if force else find_skim(cfg.name, sample, required, ff_sig)
        if info is None:
            to_build.append(sample)
        else:
            skims[sample.name] = info

    if to_build:
        jobs = [(cfg.name, sample, required, cfg.fake_factors, ff_sig)
                for sample in to_build]
        for info in run_parallel(
            build_skim, jobs, workers=workers, size_of=lambda j: j[1].size
        ):
            skims[info.sample] = info
            if on_progress is not None:
                on_progress(info)
    return skims


def prune_skims(analysis_name: str, samples: list[Sample], required: frozenset[str],
                ff_sig: dict | None = None) -> int:
    """Keep one skim per sample (newest covering `required`, else newest valid);
    delete the rest. Returns bytes freed."""
    freed = 0
    for sample in samples:
        directory = skim_dir(analysis_name, sample)
        if not directory.is_dir():
            continue
        src_sig = file_signature(sample.path)
        covering: list[tuple[float, Path]] = []
        valid: list[tuple[float, Path]] = []
        for manifest_path in directory.glob("*.json"):
            manifest = _load_manifest(manifest_path)
            if manifest is None or manifest["src"] != src_sig:
                continue
            entry = (float(manifest.get("created", 0.0)), manifest_path)
            valid.append(entry)
            if (required <= set(manifest["covers"]) | set(manifest["columns"])
                    and manifest.get("fake_factors") == ff_sig):
                covering.append(entry)
        pool = covering or valid
        keep_key = max(pool)[1].stem if pool else None

        for path in directory.iterdir():
            if path.name.split(".")[0] == keep_key:
                continue
            freed += path.stat().st_size
            path.unlink()
    return freed


def read_skim(info: SkimInfo, columns: list[str] | None = None, filter_expr=None):
    """Read a skim as a pyarrow Table with optional filter pushdown."""
    import pyarrow.parquet as pq

    use_cols = None
    if columns is not None:
        use_cols = [c for c in columns if c in info.columns]
    return pq.read_table(info.path, columns=use_cols, filters=filter_expr)
