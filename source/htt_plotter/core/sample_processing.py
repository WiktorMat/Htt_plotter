from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from htt_plotter.io.data_access import DataAccess
from htt_plotter.physics.weights import compute_mc_weight
from htt_plotter.plotting.accumulate import add_histogram
from htt_plotter.selection.selection import make_arrow_filter, selection_columns_used


def to_numpy(batch: Any, name: str) -> np.ndarray:
    col = batch.column(batch.schema.get_field_index(name))
    return col.to_numpy(zero_copy_only=False)


SampleResult = dict[str, Any]


def process_sample(
    payload: dict[str, Any],
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mc_weight_cache: dict | None = None,
) -> SampleResult:
    """Process a single sample (one index item) and return partial histograms.

    Designed to be picklable for multiprocessing: only `payload` is required.
    In serial mode, `progress_callback` and `mc_weight_cache` can be provided.
    """

    item = payload["item"]
    sample_config = payload["sample_config"]
    params = payload["params"]
    plotter_config = payload["plotter_config"]
    variable_config = payload["variable_config"]
    sample_to_process = payload["sample_to_process"]

    contr_name: list[str] = payload.get("contr_name") or []
    resolution_pairs = [tuple(x) for x in (payload.get("resolution_pairs") or [])]

    control_edges: dict[str, Any] = payload.get("control_edges") or {}
    resolution_edges: dict[tuple[str, str], Any] = payload.get("resolution_edges") or {}

    do_control = bool(payload.get("do_control", True))
    do_resolution = bool(payload.get("do_resolution", True))
    do_mc_data = bool(payload.get("do_mc_data", True))

    asym_cfg = payload.get("asym_cfg") or {}

    runtime_cfg = plotter_config.get("plotter_runtime", {}) or {}
    prefetch_batches = int(runtime_cfg.get("io_prefetch_batches", 4) or 0)
    if prefetch_batches < 0:
        prefetch_batches = 0

    data_access = DataAccess(
        Path(payload["project_root"]),
        sample_config,
        log_every_files=0,
    )

    sample = item["sample"]
    kind = item.get("kind", "mc")
    schema = set(item.get("schema", set()))

    process = sample_to_process.get(sample, "unknown")

    present_control = [v for v in contr_name if v in schema]
    present_pairs = [(c, r) for (c, r) in resolution_pairs if c in schema and r in schema]

    selection_cfg = plotter_config.get("selection", {}) or {}
    selection_cols = selection_columns_used(selection_cfg)

    needed: set[str] = set()
    if do_control:
        needed |= set(present_control)
    if do_resolution:
        for c, r in present_pairs:
            needed.add(c)
            needed.add(r)
    if do_mc_data:
        needed |= set(present_control)
        needed.add("os")
        needed.add("trg_singlemuon")
        needed.add("trg_mt_cross")
        needed.add("weight")

    needed |= {c for c in selection_cols if c in schema}

    columns = sorted(c for c in needed if c in schema)

    filter_expr = make_arrow_filter(plotter_config, schema)

    if kind == "data":
        mc_weight = 1.0
    else:
        params_key = (sample_config.get(sample) or {}).get("params_key", sample)
        mc_weight = compute_mc_weight(params_key, params, cache=mc_weight_cache or {})

    control_partial: dict[str, dict[str, np.ndarray]] = {}
    resolution_partial: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    agreement_partial: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    asymmetry_partial: dict[str, dict[str, np.ndarray]] = {}

    progress_interval_s = float(payload.get("progress_interval_s", 10.0) or 10.0)

    for batch in data_access.iter_batches(
        item,
        columns=columns,
        filter_expr=filter_expr,
        progress_callback=progress_callback,
        progress_interval_s=progress_interval_s,
        prefetch_batches=prefetch_batches,
    ):
        if do_control and present_control:
            for var in present_control:
                values = to_numpy(batch, var)
                mask = np.isfinite(values)
                if not np.any(mask):
                    continue

                edges = control_edges.get(var)
                if edges is None or len(edges) < 2:
                    continue

                counts, _ = np.histogram(values[mask], bins=edges)

                by_var = control_partial.setdefault(var, {})
                if process not in by_var:
                    by_var[process] = np.zeros(len(edges) - 1, dtype=float)
                by_var[process] += counts

        if do_resolution and present_pairs:
            for c, r in present_pairs:
                cv = to_numpy(batch, c)
                rv = to_numpy(batch, r)

                mask = np.isfinite(cv) & np.isfinite(rv) & (cv != 0)
                if not np.any(mask):
                    continue

                var_cfg = variable_config.get(r, {})
                is_angle = var_cfg.get("type") == "angle"
                relative = var_cfg.get("relative_resolution", True)

                if is_angle or not relative:
                    resolution = rv[mask] - cv[mask]
                    resolution = (resolution + np.pi) % (2 * np.pi) - np.pi
                else:
                    resolution = (rv[mask] - cv[mask]) / cv[mask]

                edges = resolution_edges.get((c, r))
                if edges is None or len(edges) < 2:
                    continue

                counts, _ = np.histogram(resolution, bins=edges)

                by_pair = resolution_partial.setdefault((c, r), {})
                if process not in by_pair:
                    by_pair[process] = np.zeros(len(edges) - 1, dtype=float)
                by_pair[process] += counts

        if do_mc_data and present_control and ("os" in schema) and ("os" in batch.schema.names):
            os_flag = to_numpy(batch, "os")

            for var in present_control:
                values = to_numpy(batch, var)
                trg_single = to_numpy(batch, "trg_singlemuon")
                trg_cross = to_numpy(batch, "trg_mt_cross")
                trigger_mask = (trg_single == 1) | (trg_cross == 1)
                weights = to_numpy(batch, "weight")

                by_var = agreement_partial.setdefault(var, {"OS": {}, "SS": {}})

                for region_name, region_mask in {"OS": os_flag == 1, "SS": os_flag == 0}.items():
                    mask = region_mask & np.isfinite(values) # & trigger_mask
                    if not np.any(mask):
                        continue

                    edges = control_edges.get(var)
                    if edges is None or len(edges) < 2:
                        continue

                    raw_counts, _ = np.histogram(values[mask], bins=edges, weights=weights[mask])
                    raw_counts = raw_counts.astype(float)
                    counts = raw_counts * mc_weight
                    sumw2 = raw_counts * (mc_weight**2)

                    region = by_var[region_name]
                    if process not in region:
                        region[process] = {
                            "counts": np.zeros(len(edges) - 1, dtype=float),
                            "sumw2": np.zeros(len(edges) - 1, dtype=float),
                        }

                    region[process]["counts"] += counts
                    region[process]["sumw2"] += sumw2

        if asym_cfg.get("enable", False):
            column = asym_cfg.get("column")
            if column in schema and column in batch.schema.names:
                values = to_numpy(batch, column)

                if (
                    asym_cfg.get("cp_weights", True)
                    and ("wt_cp_sm" in batch.schema.names)
                    and ("wt_cp_ps" in batch.schema.names)
                ):
                    w_even = to_numpy(batch, "wt_cp_sm")
                    w_odd = to_numpy(batch, "wt_cp_ps")
                else:
                    w_even = np.ones_like(values)
                    w_odd = np.ones_like(values)

                bins = int(asym_cfg.get("bins", 8))
                range_ = tuple(asym_cfg.get("range", (0, 2 * np.pi)))

                hist_even, _ = np.histogram(values, bins=bins, range=range_, weights=w_even)
                hist_odd, _ = np.histogram(values, bins=bins, range=range_, weights=w_odd)

                if column not in asymmetry_partial:
                    asymmetry_partial[column] = {
                        "even": np.zeros_like(hist_even, dtype=float),
                        "odd": np.zeros_like(hist_odd, dtype=float),
                    }
                asymmetry_partial[column]["even"] += hist_even
                asymmetry_partial[column]["odd"] += hist_odd

    return {
        "sample": sample,
        "kind": kind,
        "process": process,
        "control": control_partial,
        "resolution": resolution_partial,
        "agreement": agreement_partial,
        "asymmetry": asymmetry_partial,
    }


def merge_sample_result(
    result: SampleResult,
    *,
    control_hists: dict[str, dict[str, np.ndarray]] | None = None,
    resolution_hists: dict[tuple[str, str], dict[str, np.ndarray]] | None = None,
    agreement: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] | None = None,
    asymmetry_buffer: dict[str, dict[str, list[np.ndarray]]] | None = None,
    process_kinds: dict[str, str] | None = None,
) -> None:
    """Merge one SampleResult into global containers."""

    process = str(result.get("process", "unknown"))
    kind = str(result.get("kind", "mc"))

    if process_kinds is not None:
        if kind == "data" or process not in process_kinds:
            process_kinds[process] = kind

    if control_hists is not None:
        for var, by_process in (result.get("control") or {}).items():
            if var not in control_hists:
                continue
            for proc_name, counts in by_process.items():
                add_histogram(control_hists[var], proc_name, np.asarray(counts, dtype=float))

    if resolution_hists is not None:
        for pair_key, by_process in (result.get("resolution") or {}).items():
            pair = tuple(pair_key)
            if pair not in resolution_hists:
                continue
            for proc_name, counts in by_process.items():
                add_histogram(resolution_hists[pair], proc_name, np.asarray(counts, dtype=float))

    if agreement is not None:
        for var, regions in (result.get("agreement") or {}).items():
            if var not in agreement:
                continue
            for region_name in ("OS", "SS"):
                region = (regions or {}).get(region_name, {})
                for proc_name, hist in region.items():
                    target = agreement[var][region_name]
                    if proc_name not in target:
                        target[proc_name] = {
                            "counts": np.zeros_like(np.asarray(hist["counts"], dtype=float)),
                            "sumw2": np.zeros_like(np.asarray(hist["sumw2"], dtype=float)),
                        }
                    target[proc_name]["counts"] += np.asarray(hist["counts"], dtype=float)
                    target[proc_name]["sumw2"] += np.asarray(hist["sumw2"], dtype=float)

    if asymmetry_buffer is not None:
        for column, data in (result.get("asymmetry") or {}).items():
            buf = asymmetry_buffer.setdefault(column, {"even": [], "odd": []})
            buf["even"].append(np.asarray(data.get("even", []), dtype=float))
            buf["odd"].append(np.asarray(data.get("odd", []), dtype=float))
