#!/usr/bin/env python
"""Append per-event ABCD QCD weights to WHAM input parquet files."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from wham.cli import _load, _resolve_config
from wham.config import AnalysisConfig, Sample, sample_scale
from wham.expr import parse
from wham.fill import build_fill_spec, fill_all
from wham.qcd import abcd_transfer_factors
from wham.skim import ensure_skims


W_ABCD = "w_abcd"


def _select_variable(cfg: AnalysisConfig, requested: str | None) -> str:
    if requested is not None:
        if requested not in cfg.plots.datamc:
            choices = ", ".join(cfg.plots.datamc)
            raise ValueError(f"--var must be one of plots.datamc: {choices}")
        return requested
    if len(cfg.plots.datamc) == 1:
        return cfg.plots.datamc[0]
    choices = ", ".join(cfg.plots.datamc)
    raise ValueError(
        "ABCD TF is binned in one datamc variable; choose one with --var "
        f"(configured: {choices})"
    )


class _Columns:
    def __init__(self, table):
        self._table = table
        self._cache: dict[str, np.ndarray] = {}
        self.names = set(table.column_names)

    def get(self, name: str) -> np.ndarray:
        arr = self._cache.get(name)
        if arr is None:
            arr = self._table.column(name).to_numpy(zero_copy_only=False)
            self._cache[name] = arr
        return arr

    def eval(self, source: str) -> np.ndarray:
        parsed = parse(source)
        return parsed.evaluate({c: self.get(c) for c in parsed.columns})


def _eval_if_available(cols: _Columns, source: str | None) -> np.ndarray | None:
    if source is None:
        return None
    parsed = parse(source)
    if not parsed.columns <= cols.names:
        return None
    return parsed.evaluate({c: cols.get(c) for c in parsed.columns})


def _variable_bins(cols: _Columns, var: str, vcfg: dict, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unroll = vcfg.get("unroll")
    if unroll is None:
        col = vcfg.get("column") or var
        if col not in cols.names:
            raise ValueError(f"source file is missing variable column '{col}'")
        values = cols.get(col)
        idx = np.searchsorted(edges, values, side="right") - 1
        valid = np.isfinite(values) & (idx >= 0) & (idx < len(edges) - 1)
        return idx, valid

    xcol, ycol = unroll["x_column"], unroll["y_column"]
    if xcol not in cols.names or ycol not in cols.names:
        raise ValueError(f"source file is missing unroll columns '{xcol}'/'{ycol}'")
    xe = np.asarray(unroll["x_edges"], dtype=float)
    ye = np.asarray(unroll["y_edges"], dtype=float)
    ix = np.searchsorted(xe, cols.get(xcol), side="right") - 1
    iy = np.searchsorted(ye, cols.get(ycol), side="right") - 1
    nx, ny = len(xe) - 1, len(ye) - 1
    idx = ix + nx * iy
    valid = (
        np.isfinite(cols.get(xcol))
        & np.isfinite(cols.get(ycol))
        & (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
        & (idx >= 0)
        & (idx < nx * ny)
    )
    return idx, valid


def _sample_process_mask(cfg: AnalysisConfig, sample: Sample, cols: _Columns) -> np.ndarray:
    mask = _eval_if_available(cols, cfg.selection)
    if mask is None:
        raise ValueError(f"{sample.path} lacks columns needed by selection")
    mask = mask.astype(bool)

    trigger = _eval_if_available(cols, cfg.trigger)
    if trigger is not None:
        mask &= trigger.astype(bool)

    cut = cfg.processes[sample.process].cut
    if cut is not None:
        cut_mask = _eval_if_available(cols, cut)
        if cut_mask is not None:
            mask &= cut_mask.astype(bool)
    return mask


def _sample_nominal_weights(cfg: AnalysisConfig, sample: Sample, cols: _Columns) -> np.ndarray:
    n = next(iter(cols._cache.values())).shape[0] if cols._cache else cols._table.num_rows
    if sample.kind == "data":
        return np.ones(n)
    if parse(cfg.weight).columns <= cols.names:
        return cols.eval(cfg.weight).astype(float) * sample_scale(sample, cfg.lumi)
    return np.full(n, sample_scale(sample, cfg.lumi))


def _abcd_weights_for_file(
    cfg: AnalysisConfig,
    samples: list[Sample],
    table,
    *,
    var: str,
    vcfg: dict,
    edges: np.ndarray,
    transfer_factors: np.ndarray,
) -> np.ndarray:
    cols = _Columns(table)
    idx, valid_bin = _variable_bins(cols, var, vcfg, edges)

    os_mask = _eval_if_available(cols, cfg.qcd.os)
    iso_mask = _eval_if_available(cols, cfg.qcd.iso)
    if os_mask is None or iso_mask is None:
        raise ValueError("source file lacks columns needed by qcd.os/qcd.iso")

    ss_iso = (~os_mask.astype(bool)) & iso_mask.astype(bool)
    out = np.zeros(table.num_rows, dtype=float)
    for sample in samples:
        base = _sample_process_mask(cfg, sample, cols)
        mask = base & ss_iso & valid_bin
        if not np.any(mask):
            continue
        weights = _sample_nominal_weights(cfg, sample, cols)
        sign = 1.0 if sample.kind == "data" else -1.0
        out[mask] += sign * weights[mask] * transfer_factors[idx[mask]]
    return out


def _write_table(path: Path, table, weights: np.ndarray, *, column: str, dry_run: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if column in table.column_names:
        table = table.set_column(
            table.column_names.index(column), column, pa.array(weights, type=pa.float64())
        )
    else:
        table = table.append_column(column, pa.array(weights, type=pa.float64()))

    if dry_run:
        return
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{int(time.time())}")
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, path)


def run(
    config: str,
    *,
    var: str | None,
    workers: int,
    force_skims: bool,
    dry_run: bool,
) -> int:
    import pyarrow.parquet as pq

    cfg, samples = _load(str(_resolve_config(config)))
    if cfg.qcd.method != "abcd":
        raise ValueError("this script currently implements qcd.method=abcd only")
    var = _select_variable(cfg, var)

    skims = ensure_skims(cfg, samples, workers=workers, force=force_skims)
    hists = fill_all(
        cfg,
        samples,
        skims,
        families=["datamc"],
        only_vars=(var,),
        workers=workers,
        use_cache=not force_skims,
        write_cache=False,
        sidecars=False,
    )
    hist = hists[("datamc", var)]
    transfer_factors = abcd_transfer_factors(cfg, hist)
    edges = np.asarray(hist.axes[-1].edges, dtype=float)
    spec = build_fill_spec(cfg, families=["datamc"], only_vars=(var,))
    vcfg = dict(spec.datamc[0][1])

    by_name: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.kind in ("data", "mc"):
            by_name.setdefault(sample.name, []).append(sample)

    for sample_name, entries in sorted(by_name.items()):
        path = entries[0].path
        table = pq.read_table(path)
        weights = _abcd_weights_for_file(
            cfg,
            entries,
            table,
            var=var,
            vcfg=vcfg,
            edges=edges,
            transfer_factors=transfer_factors,
        )
        _write_table(path, table, weights, column=W_ABCD, dry_run=dry_run)
        total = float(weights.sum())
        changed = int(np.count_nonzero(weights))
        action = "would update" if dry_run else "updated"
        print(f"{action} {sample_name}: {changed:,}/{table.num_rows:,} nonzero, sum={total:.6g}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append a w_abcd column derived from WHAM's ABCD QCD estimate."
    )
    parser.add_argument("config", help="WHAM analysis YAML path or bare Configurations/<name>")
    parser.add_argument("--var", help="datamc variable whose bins define the ABCD transfer factor")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force-skims", action="store_true", help="rebuild skims before filling")
    parser.add_argument("--dry-run", action="store_true", help="compute weights without writing parquet files")
    args = parser.parse_args(argv)
    try:
        return run(
            args.config,
            var=args.var,
            workers=args.workers,
            force_skims=args.force_skims,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
