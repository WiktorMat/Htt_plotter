"""Signal-region QCD estimate from two methods (ABCD vs per-event BDT FF),
overlaid with an FF/ABCD ratio panel.

Both estimates come from the standard datamc fills of two config variants
that differ only in qcd.method; the QCD process slot of each histogram
already holds the finished estimate. The two methods share the OS anti-iso
data, so the ratio errors (combined as if uncorrelated) are conservative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from wham.config import AnalysisConfig
from wham.render.common import cms_label, save, slice_1d, step_with_band, var_label


def _qcd_curve(h, qcd_proc: str) -> tuple[np.ndarray, np.ndarray]:
    view = slice_1d(h, qcd_proc, "OS_iso").view()
    return view["value"], np.sqrt(np.maximum(view["variance"], 0.0))


def render_qcdcompare(
    cfg: AnalysisConfig, hists_abcd: dict, hists_ff: dict, outdir: Path,
    only_vars=None, console=None,
) -> None:
    qcd_proc = cfg.qcd_process()
    if qcd_proc is None:
        return
    qcd_color = cfg.processes[qcd_proc].color

    for (family, var), h_ff in sorted(hists_ff.items()):
        if family != "datamc" or (only_vars and var not in only_vars):
            continue
        h_abcd = hists_abcd.get((family, var))
        if h_abcd is None or "OS_iso" not in list(h_ff.axes["region"]):
            continue

        edges = h_ff.axes[-1].edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        abcd, abcd_unc = _qcd_curve(h_abcd, qcd_proc)
        ff, ff_unc = _qcd_curve(h_ff, qcd_proc)

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        rax = fig.add_subplot(gs[1], sharex=ax)

        step_with_band(ax, edges, abcd, abcd_unc, "black",
                       f"ABCD ({abcd.sum():,.0f} ev)")
        step_with_band(ax, edges, ff, ff_unc, qcd_color,
                       f"BDT FF ({ff.sum():,.0f} ev)")

        ax.set_xlabel("")
        ax.set_ylabel("QCD events (signal region)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=17)
        ax.tick_params(labelbottom=False)
        cms_label(ax, cfg)

        # ---- ratio panel
        safe = np.where(abcd > 0, abcd, np.nan)
        ratio = ff / safe
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.sqrt((ff_unc / np.where(ff > 0, ff, np.nan)) ** 2
                          + (abcd_unc / safe) ** 2)
        ratio_unc = np.where(np.isfinite(rel), ratio * rel, 0.0)

        rax.axhline(1.0, linestyle="--", color="black", linewidth=1)
        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt="o",
                     color=qcd_color, markersize=5)
        finite = np.isfinite(ratio)
        top = 1.3 * float(np.percentile(ratio[finite], 90)) if finite.any() else 2.0
        rax.set_ylim(0, max(top, 1.5))
        rax.set_ylabel("FF / ABCD")
        rax.set_xlabel(var_label(cfg, var))

        save(fig, outdir / "qcdcompare", var, console)
