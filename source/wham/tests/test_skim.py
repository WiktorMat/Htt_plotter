from __future__ import annotations

import os

import pyarrow.parquet as pq

from wham.config import discover_samples, load_config
from wham.skim import build_skim, ensure_skims, find_skim, prune_skims


def test_build_find_reuse(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    required = cfg.required_columns()
    sample = samples[0]

    assert find_skim(cfg.name, sample, required) is None
    info = build_skim(cfg.name, sample, required)
    assert info.path.is_file()
    assert info.rows == 4000
    # data sample lacks wt_cp_* -> intersection recorded, still covered
    assert info.columns <= required

    found = find_skim(cfg.name, sample, required)
    assert found is not None and found.path == info.path

    # subset of required columns reuses the same skim
    smaller = frozenset(list(required)[:3])
    assert find_skim(cfg.name, sample, smaller) is not None

    # superset forces a rebuild
    bigger = required | {"extra_col"}
    assert find_skim(cfg.name, sample, bigger) is None


def test_invalidation_on_source_change(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    sample = samples[0]
    required = cfg.required_columns()

    build_skim(cfg.name, sample, required)
    assert find_skim(cfg.name, sample, required) is not None

    os.utime(sample.path, ns=(sample.mtime_ns + 10**9, sample.mtime_ns + 10**9))
    assert find_skim(cfg.name, sample, required) is None


def test_ensure_and_prune(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    required = cfg.required_columns()

    skims = ensure_skims(cfg, samples, workers=1)
    assert set(skims) == {s.name for s in samples}

    # second call builds nothing (all found)
    built = []
    ensure_skims(cfg, samples, workers=1, on_progress=lambda i: built.append(i))
    assert built == []

    # stale skim from an older column set gets pruned; the covering one survives
    build_skim(cfg.name, samples[0], frozenset({"pt_1"}))
    freed = prune_skims(cfg.name, samples, required)
    assert freed > 0
    assert find_skim(cfg.name, samples[0], required) is not None
    assert find_skim(cfg.name, samples[0], frozenset({"pt_1"})) is not None  # covered by big skim


def test_skim_row_content_identical(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    sample = samples[0]
    info = build_skim(cfg.name, sample, frozenset({"pt_1", "weight"}))

    skimmed = pq.read_table(info.path)
    original = pq.read_table(sample.path, columns=sorted(skimmed.column_names))
    assert skimmed.sort_by("pt_1").equals(original.sort_by("pt_1"))
