"""QCD multijet estimation on filled histograms.

Faithful port of the old backgrounds/qcd.py math:

ss method:
    QCD_OS = clip(data_SS - mc_SS, 0) * ff           sumw2 -> (s2_data+s2_mc) * ff^2
abcd method (per bin):
    tf        = clip(data-mc, 0)_OS_antiiso / clip(data-mc, 0)_SS_antiiso
    QCD_OS    = clip(data-mc, 0)_SS_iso * tf          sumw2_SS_iso * tf^2
    QCD_SS    = clip(data-mc, 0)_SS_iso               (kept for inspection)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wham.config import AnalysisConfig


def _region_sums(h: Any, region: str, *, processes: list[str], data_proc: str,
                 qcd_proc: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(data, data_sumw2, mc, mc_sumw2) along the variable axis for one region."""
    nbins = h.axes[-1].size
    data = np.zeros(nbins)
    data_w2 = np.zeros(nbins)
    mc = np.zeros(nbins)
    mc_w2 = np.zeros(nbins)
    for proc in processes:
        view = h[{"process": proc, "region": region, "variation": "nominal"}].view()
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


def _set_qcd(h: Any, region: str, qcd_proc: str, counts: np.ndarray, sumw2: np.ndarray) -> None:
    idx_p = list(h.axes["process"]).index(qcd_proc)
    idx_r = list(h.axes["region"]).index(region)
    view = h.view()
    view[idx_p, idx_r, 0, :]["value"] = counts
    view[idx_p, idx_r, 0, :]["variance"] = sumw2


def estimate_qcd(cfg: AnalysisConfig, datamc_hists: dict[Any, Any]) -> None:
    """Fill the QCD process slot of every datamc histogram, in place."""
    qcd_proc = cfg.qcd_process()
    data_proc = cfg.data_process()
    if qcd_proc is None or data_proc is None:
        return

    processes = list(cfg.processes.keys())
    method = cfg.qcd.method

    for (_family, _var), h in datamc_hists.items():
        regions = list(h.axes["region"])

        if method == "abcd":
            if not {"OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso"} <= set(regions):
                continue
            qcd = {}
            for region in ("SS_iso", "OS_antiiso", "SS_antiiso"):
                data, data_w2, mc, mc_w2 = _region_sums(
                    h, region, processes=processes, data_proc=data_proc, qcd_proc=qcd_proc
                )
                qcd[region] = (np.maximum(data - mc, 0.0), data_w2 + mc_w2)

            tf = _safe_ratio(qcd["OS_antiiso"][0], qcd["SS_antiiso"][0])
            counts = np.maximum(qcd["SS_iso"][0] * tf, 0.0)
            sumw2 = qcd["SS_iso"][1] * tf**2
            _set_qcd(h, "OS_iso", qcd_proc, counts, sumw2)
            _set_qcd(h, "SS_iso", qcd_proc, qcd["SS_iso"][0], qcd["SS_iso"][1])
        else:
            if not {"OS", "SS"} <= set(regions):
                continue
            data, data_w2, mc, mc_w2 = _region_sums(
                h, "SS", processes=processes, data_proc=data_proc, qcd_proc=qcd_proc
            )
            ss_counts = np.maximum(data - mc, 0.0)
            ss_sumw2 = data_w2 + mc_w2
            ff = float(cfg.qcd.ff)
            _set_qcd(h, "OS", qcd_proc, np.maximum(ss_counts * ff, 0.0), ss_sumw2 * ff**2)
            _set_qcd(h, "SS", qcd_proc, ss_counts, ss_sumw2)
