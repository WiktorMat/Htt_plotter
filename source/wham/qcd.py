"""QCD multijet estimation on filled histograms.

Faithful port of the old backgrounds/qcd.py math:

ss method:
    QCD_OS = clip(data_SS - mc_SS, 0) * ff           sumw2 -> (s2_data+s2_mc) * ff^2
abcd method (per bin):
    tf        = clip(data-mc, 0)_OS_antiiso / clip(data-mc, 0)_SS_antiiso
    QCD_OS    = clip(data-mc, 0)_SS_iso * tf          sumw2_SS_iso * tf^2
    QCD_SS    = clip(data-mc, 0)_SS_iso               (kept for inspection)
ff method (per bin):
    QCD_OS_iso = clip(data-mc, 0)_OS_antiiso
    where the OS_antiiso fills already carry the per-event qcd.ff_weight,
    so the subtraction removes genuine-tau MC promoted by the same weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wham.config import AnalysisConfig


def region_sums(h: Any, region: str, *, processes: list[str], data_proc: str,
                qcd_proc: str | None, variation: str = "nominal",
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(data, data_sumw2, mc, mc_sumw2) along the variable axis for one region."""
    nbins = h.axes[-1].size
    data = np.zeros(nbins)
    data_w2 = np.zeros(nbins)
    mc = np.zeros(nbins)
    mc_w2 = np.zeros(nbins)
    for proc in processes:
        view = h[{"process": proc, "region": region, "variation": variation}].view()
        if proc == data_proc:
            data += view["value"]
            data_w2 += view["variance"]
        elif proc != qcd_proc:
            mc += view["value"]
            mc_w2 += view["variance"]
    return data, data_w2, mc, mc_w2


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _set_qcd(h: Any, region: str, qcd_proc: str, counts: np.ndarray, sumw2: np.ndarray,
             variation: str = "nominal") -> None:
    idx_p = list(h.axes["process"]).index(qcd_proc)
    idx_r = list(h.axes["region"]).index(region)
    idx_v = list(h.axes["variation"]).index(variation)
    view = h.view()
    view[idx_p, idx_r, idx_v, :]["value"] = counts
    view[idx_p, idx_r, idx_v, :]["variance"] = sumw2


def abcd_transfer_factors(
    cfg: AnalysisConfig,
    h: Any,
    *,
    variation: str = "nominal",
) -> np.ndarray:
    """ABCD OS/SS anti-isolated transfer factors along one histogram axis.

    This is the same per-bin factor used by :func:`estimate_qcd` for
    ``qcd.method=abcd``:

        clip(data - mc, 0)_OS_antiiso / clip(data - mc, 0)_SS_antiiso
    """
    if cfg.qcd.method != "abcd":
        raise ValueError("ABCD transfer factors require qcd.method=abcd")
    data_proc = cfg.data_process()
    if data_proc is None:
        raise ValueError("ABCD transfer factors require a kind=data process")
    qcd_proc = cfg.qcd_process()

    regions = set(h.axes["region"])
    if not {"OS_antiiso", "SS_antiiso"} <= regions:
        raise ValueError("histogram does not contain ABCD anti-isolated regions")

    def qcd_counts(region: str) -> np.ndarray:
        data, _data_w2, mc, _mc_w2 = region_sums(
            h,
            region,
            processes=list(cfg.processes.keys()),
            data_proc=data_proc,
            qcd_proc=qcd_proc,
            variation=variation,
        )
        return np.maximum(data - mc, 0.0)

    return _safe_ratio(qcd_counts("OS_antiiso"), qcd_counts("SS_antiiso"))


def estimate_qcd(cfg: AnalysisConfig, datamc_hists: dict[Any, Any]) -> None:
    """Fill the QCD process slot of every datamc histogram, in place.

    Runs per variation slice: each is a complete alternative universe (see
    fill.complete_variation_slices), so e.g. a varied MC subtraction or a
    varied FF weight propagates into that variation's QCD estimate."""
    qcd_proc = cfg.qcd_process()
    data_proc = cfg.data_process()
    if qcd_proc is None or data_proc is None:
        return

    processes = list(cfg.processes.keys())
    method = cfg.qcd.method

    for (_family, _var), h in datamc_hists.items():
        regions = list(h.axes["region"])
        for variation in list(h.axes["variation"]):
            sums = {}

            def region_qcd(region: str) -> tuple[np.ndarray, np.ndarray]:
                data, data_w2, mc, mc_w2 = region_sums(
                    h, region, processes=processes, data_proc=data_proc,
                    qcd_proc=qcd_proc, variation=variation,
                )
                return np.maximum(data - mc, 0.0), data_w2 + mc_w2

            if method == "abcd":
                if not {"OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso"} <= set(regions):
                    continue
                for region in ("SS_iso", "OS_antiiso", "SS_antiiso"):
                    sums[region] = region_qcd(region)
                tf = abcd_transfer_factors(cfg, h, variation=variation)
                counts = np.maximum(sums["SS_iso"][0] * tf, 0.0)
                sumw2 = sums["SS_iso"][1] * tf**2
                _set_qcd(h, "OS_iso", qcd_proc, counts, sumw2, variation)
                _set_qcd(h, "SS_iso", qcd_proc, *sums["SS_iso"], variation)
            elif method == "ff":
                if not {"OS_iso", "OS_antiiso"} <= set(regions):
                    continue
                counts, sumw2 = region_qcd("OS_antiiso")
                _set_qcd(h, "OS_iso", qcd_proc, counts, sumw2, variation)
            else:
                if not {"OS", "SS"} <= set(regions):
                    continue
                ss_counts, ss_sumw2 = region_qcd("SS")
                ff = float(cfg.qcd.ff)
                _set_qcd(h, "OS", qcd_proc, np.maximum(ss_counts * ff, 0.0),
                         ss_sumw2 * ff**2, variation)
                _set_qcd(h, "SS", qcd_proc, ss_counts, ss_sumw2, variation)
