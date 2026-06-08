from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from htt_plotter.selection.selection import ARROW_OPS

TAU_ID_BRANCH = "idDeepTau2018v2p5VSjet_2"
VVVLOOSE_WP = 1
VTIGHT_WP = 5


def qcd_method(config: dict | None) -> str:
    cfg = config or {}
    method = cfg.get("qcd_method")
    if method is None:
        method = (cfg.get("plotter_runtime") or {}).get("qcd_method", "ss")
    method = str(method).strip().lower()
    return method or "ss"


def qcd_region_names(method: str) -> list[str]:
    if str(method).strip().lower() == "abcd":
        return ["OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso"]
    return ["OS", "SS"]


def qcd_region_columns(region_cfg: dict[str, Any] | None) -> set[str]:
    columns: set[str] = set()
    for region_def in (region_cfg or {}).values():
        if not isinstance(region_def, dict):
            continue
        columns.update(region_def.keys())
    return columns


def _normalize_conditions(rule: Any) -> list[tuple[str, Any]]:
    if isinstance(rule, tuple) and len(rule) == 2 and isinstance(rule[0], str):
        return [(str(rule[0]), rule[1])]
    if isinstance(rule, list):
        out: list[tuple[str, Any]] = []
        for item in rule:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                out.append((str(item[0]), item[1]))
            else:
                raise ValueError(f"Invalid QCD region condition: {item}")
        return out
    raise ValueError(f"Invalid QCD region rule: {rule}")


def _apply_condition(values: np.ndarray, op: str, target: Any) -> np.ndarray:
    func = ARROW_OPS.get(op)
    if func is None:
        raise ValueError(f"Unknown QCD region operator: {op}")
    return func(values, target)


def qcd_region_mask(batch, region_def: dict[str, Any] | None) -> np.ndarray:
    if not region_def:
        return np.ones(batch.num_rows, dtype=bool)

    mask = np.ones(batch.num_rows, dtype=bool)
    for column, rule in region_def.items():
        conditions = _normalize_conditions(rule)
        values = batch.column(batch.schema.get_field_index(column)).to_numpy(zero_copy_only=False)
        column_mask = np.ones_like(values, dtype=bool)
        for op, target in conditions:
            column_mask &= _apply_condition(values, op, target)
        mask &= column_mask

    return mask


def _region_keys(method: str) -> tuple[str, str, str | None, str | None]:
    if str(method).strip().lower() == "abcd":
        return "OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso"
    return "OS", "SS", None, None


def _sum_region_components(
    region: dict,
    sample_kinds: dict[str, str],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    data_counts = None
    data_sumw2 = None
    mc_counts = None
    mc_sumw2 = None

    for name, hist in (region or {}).items():
        counts = np.asarray(hist.get("counts", 0.0), dtype=float)
        sumw2 = np.asarray(hist.get("sumw2", counts), dtype=float)

        if sample_kinds.get(name, "mc") == "data":
            if data_counts is None:
                data_counts = counts.copy()
                data_sumw2 = sumw2.copy()
            else:
                data_counts += counts
                data_sumw2 += sumw2
        elif str(name).lower() != "qcd":
            if mc_counts is None:
                mc_counts = counts.copy()
                mc_sumw2 = sumw2.copy()
            else:
                mc_counts += counts
                mc_sumw2 += sumw2

    return data_counts, data_sumw2, mc_counts, mc_sumw2


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=ratio, where=denominator != 0)
    return ratio


def _clip_non_negative(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def add_qcd_from_ss(
    histograms: dict,
    config: dict,
    sample_kinds: dict[str, str] | None = None,
) -> None:
    """Compute QCD from SS and, optionally, an ABCD transfer factor.

    Enabled by config["add_qcd_from_ss"]. Stores the derived component under key "QCD" in both
    regions. In SS mode this is direct SS->OS propagation. In ABCD mode the SS estimate is scaled
    by the transfer factor measured in antiiso.
    """

    if not config.get("add_qcd_from_ss", False):
        return

    sample_kinds = sample_kinds or {}
    method = qcd_method(config)
    os_key, ss_key, os_anti_key, ss_anti_key = _region_keys(method)

    if os_key not in histograms or ss_key not in histograms:
        return

    ss = histograms[ss_key]
    os = histograms[os_key]

    data_counts_ss, data_sumw2_ss, mc_counts_ss, mc_sumw2_ss = _sum_region_components(ss, sample_kinds)
    if data_counts_ss is None:
        print(f"[WARN] add_qcd_from_ss enabled but no data samples found in {ss_key}")
        return

    if mc_counts_ss is None:
        print(f"[WARN] add_qcd_from_ss enabled but no MC samples found in {ss_key}")
        return

    qcd_counts_ss = data_counts_ss - mc_counts_ss
    qcd_counts_ss = _clip_non_negative(qcd_counts_ss)
    qcd_sumw2_ss = data_sumw2_ss + mc_sumw2_ss

    if method == "abcd":
        if os_anti_key not in histograms or ss_anti_key not in histograms:
            print(
                f"[WARN] add_qcd_from_ss enabled but missing antiiso regions for ABCD: {os_anti_key}, {ss_anti_key}"
            )
            return

        data_counts_os, _, mc_counts_os, _ = _sum_region_components(os, sample_kinds)
        os_anti = histograms[os_anti_key]
        ss_anti = histograms[ss_anti_key]

        data_counts_os_anti, _, mc_counts_os_anti, _ = _sum_region_components(os_anti, sample_kinds)
        data_counts_ss_anti, _, mc_counts_ss_anti, _ = _sum_region_components(ss_anti, sample_kinds)

        if (
            data_counts_os is None
            or data_counts_os_anti is None
            or data_counts_ss_anti is None
            or mc_counts_os is None
            or mc_counts_os_anti is None
            or mc_counts_ss_anti is None
        ):
            print("[WARN] add_qcd_from_ss enabled but ABCD antiiso yields are incomplete")
            return

        qcd_counts_os = data_counts_os - mc_counts_os
        qcd_counts_os_anti = data_counts_os_anti - mc_counts_os_anti
        qcd_counts_ss_anti = data_counts_ss_anti - mc_counts_ss_anti
        qcd_counts_os = _clip_non_negative(qcd_counts_os)
        qcd_counts_os_anti = _clip_non_negative(qcd_counts_os_anti)
        qcd_counts_ss_anti = _clip_non_negative(qcd_counts_ss_anti)
        tf = _safe_ratio(qcd_counts_os_anti, qcd_counts_ss_anti)
        qcd_counts_os_iso = qcd_counts_ss * tf
        qcd_counts_os_iso = _clip_non_negative(qcd_counts_os_iso)
        qcd_sumw2_os_iso = qcd_sumw2_ss * (tf ** 2)

        print(
            f"[DEBUG QCD ABCD] {os_key}: data={float(np.sum(data_counts_os)):.3f} mc={float(np.sum(mc_counts_os)):.3f} qcd={float(np.sum(qcd_counts_os)):.3f}"
        )
        print(
            f"[DEBUG QCD ABCD] {ss_key}: data={float(np.sum(data_counts_ss)):.3f} mc={float(np.sum(mc_counts_ss)):.3f} qcd={float(np.sum(qcd_counts_ss)):.3f}"
        )
        print(
            f"[DEBUG QCD ABCD] {os_anti_key}: data={float(np.sum(data_counts_os_anti)):.3f} mc={float(np.sum(mc_counts_os_anti)):.3f} qcd={float(np.sum(qcd_counts_os_anti)):.3f}"
        )
        print(
            f"[DEBUG QCD ABCD] {ss_anti_key}: data={float(np.sum(data_counts_ss_anti)):.3f} mc={float(np.sum(mc_counts_ss_anti)):.3f} qcd={float(np.sum(qcd_counts_ss_anti)):.3f}"
        )

        ss["QCD"] = {"counts": qcd_counts_ss, "sumw2": qcd_sumw2_ss}
        os["QCD"] = {"counts": qcd_counts_os_iso, "sumw2": qcd_sumw2_os_iso}
        return

    ss["QCD"] = {"counts": qcd_counts_ss, "sumw2": qcd_sumw2_ss}

    ff = float(config.get("qcd_ff", 1.0))
    os["QCD"] = {"counts": _clip_non_negative(qcd_counts_ss * ff), "sumw2": qcd_sumw2_ss * (ff ** 2)}


def ensure_qcd_placeholder(
    histograms: dict[str, np.ndarray],
    desired_order: list[str],
    template: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Insert an empty QCD histogram when QCD is requested in draw order."""

    out = dict(histograms)
    qcd_in_order = any(str(name).lower() == "qcd" for name in desired_order)

    if not qcd_in_order or template is None:
        return out

    if not any(str(name).lower() == "qcd" for name in out):
        out["QCD"] = np.zeros_like(template, dtype=float)

    return out
