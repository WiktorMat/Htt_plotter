"""Histogram cache (pickle) + long-format sidecar parquet export.

Cache key covers everything that changes bin contents: skim signatures,
selection/trigger/weight/qcd expressions, process->sample mapping, lumi,
sample params and the variable's binning. Labels/colors/styles do NOT
enter the key, so re-rendering never refills.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from wham import util
from wham.config import AnalysisConfig, Sample
from wham.skim import SkimInfo


def _hist_dir(analysis_name: str) -> Path:
    return util.cache_root() / "hists" / analysis_name


def cache_key(
    cfg: AnalysisConfig,
    samples: list[Sample],
    skims: dict[str, SkimInfo],
    spec: Any,
    family: str,
    name: str,
    vcfg: dict,
    *,
    extra: dict | None = None,
) -> str:
    # new optional qcd fields are dropped when unset so old keys stay valid
    qcd_payload = cfg.qcd.model_dump()
    if qcd_payload.get("ff_weight") is None:
        del qcd_payload["ff_weight"]
    payload = {
        "sources": {
            s.name: skims[s.name].src_signature for s in samples if s.name in skims
        },
        "selection": cfg.selection,
        "trigger": cfg.trigger,
        "weight": cfg.weight,
        "qcd": qcd_payload,
        "processes": {
            n: {"kind": p.kind, "samples": sorted(
                s.name for s in samples if s.process == n)}
            for n, p in cfg.processes.items()
        },
        "lumi": cfg.lumi,
        "params": {
            s.name: s.params.model_dump() for s in samples if s.params is not None
        },
        "family": family,
        "binning": {k: vcfg.get(k) for k in ("bins", "range", "kind", "relative")},
    }
    if vcfg.get("column"):
        # conditional so keys of plain (non-aliased) variables stay stable
        payload["binning"]["column"] = vcfg["column"]
    if vcfg.get("unroll"):
        # the resolved payload (sub-columns + edges), so re-binning either
        # sub-variable invalidates the unrolled histogram
        payload["binning"]["unroll"] = vcfg["unroll"]
    if cfg.fake_factors is not None:
        from wham.muffin import signature

        payload["fake_factors"] = signature(cfg.fake_factors)
    if extra:
        # e.g. CP weight columns; added conditionally so keys of histograms
        # without extras (the vast majority) stay stable.
        payload["extra"] = extra
    return util.short_key(payload)


def _path(analysis_name: str, family: str, name: str, key: str) -> Path:
    return _hist_dir(analysis_name) / f"{family}__{name}__{key}.pkl"


def load_hist(analysis_name: str, family: str, name: str, key: str):
    path = _path(analysis_name, family, name, key)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload["hist"]
    except Exception:
        return None


def save_hist(analysis_name: str, family: str, name: str, key: str, h: Any) -> Path:
    directory = _hist_dir(analysis_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path(analysis_name, family, name, key)
    with open(path, "wb") as f:
        pickle.dump({"hist": h, "family": family, "name": name, "key": key}, f)

    index = directory / "index.json"
    entries: dict[str, str] = {}
    if index.is_file():
        try:
            entries = json.loads(index.read_text())
        except json.JSONDecodeError:
            entries = {}
    entries[f"{family}__{name}"] = path.name
    index.write_text(json.dumps(entries, indent=1, sort_keys=True))
    return path


def load_all_cached(cfg: AnalysisConfig, samples: list[Sample]) -> dict[tuple[str, str], Any]:
    """Best-effort load of everything in the index (for `inspect --yields`)."""
    directory = _hist_dir(cfg.name)
    index = directory / "index.json"
    if not index.is_file():
        return {}
    try:
        entries = json.loads(index.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[tuple[str, str], Any] = {}
    for label, filename in entries.items():
        path = directory / filename
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            continue
        family, name = label.split("__", 1)
        out[(family, name)] = payload["hist"]
    return out


def write_sidecars(cfg: AnalysisConfig, hists: dict[tuple[str, str], Any]) -> None:
    """Long-format parquet per family: one row per (variable, process, region)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_family: dict[str, list[tuple[str, Any]]] = {}
    for (family, name), h in hists.items():
        by_family.setdefault(family, []).append((name, h))

    outdir = cfg.resolved_output_dir()
    for family, items in by_family.items():
        rows: dict[str, list] = {
            "plot_type": [], "variable": [], "process": [], "region": [],
            "counts": [], "sumw2": [], "bin_edges": [],
        }
        for name, h in items:
            edges = h.axes[-1].edges.tolist()
            for proc in h.axes["process"]:
                for region in h.axes["region"]:
                    view = h[{"process": proc, "region": region, "variation": "nominal"}].view()
                    if not np.any(view["value"]):
                        continue
                    rows["plot_type"].append(family)
                    rows["variable"].append(name)
                    rows["process"].append(proc)
                    rows["region"].append(region)
                    rows["counts"].append(view["value"].tolist())
                    rows["sumw2"].append(view["variance"].tolist())
                    rows["bin_edges"].append(edges)
        if not rows["variable"]:
            continue
        family_dir = outdir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table(rows)
        compression = "snappy" if pa.Codec.is_available("snappy") else "none"
        pq.write_table(table, family_dir / "histograms.parquet", compression=compression)
