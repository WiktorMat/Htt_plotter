"""FF diagnostics: anti-iso QCD (data − MC) before vs after the BDT FF weight.

The lower panel shows weighted/raw, i.e. the effective per-bin fake factor.
For a histogram of the score column itself the ratio must follow the bin
centers — a built-in closure check of the weighting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from wham.config import AnalysisConfig
from wham.qcd import region_sums
from wham.render.common import cms_label, save, step_with_band, var_label

RAW, FF = "OS_antiiso_raw", "OS_antiiso_ff"


def render_ffcheck(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    data_proc = cfg.data_process()
    qcd_proc = cfg.qcd_process()
    qcd_color = cfg.processes[qcd_proc].color if qcd_proc else "tab:olive"
    processes = list(cfg.processes.keys())

    for (family, var), h in sorted(hists.items()):
        if family != "ffcheck" or (only_vars and var not in only_vars):
            continue
        if data_proc is None or not {RAW, FF} <= set(h.axes["region"]):
            continue

        edges = h.axes[-1].edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        curves = {}
        for region in (RAW, FF):
            data, data_w2, mc, mc_w2 = region_sums(
                h, region, processes=processes, data_proc=data_proc, qcd_proc=qcd_proc
            )
            curves[region] = (np.maximum(data - mc, 0.0),
                              np.sqrt(np.maximum(data_w2 + mc_w2, 0.0)))

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        rax = fig.add_subplot(gs[1], sharex=ax)

        step_with_band(ax, edges, *curves[RAW], "black",
                       "Anti-iso data $-$ MC (raw)")
        step_with_band(ax, edges, *curves[FF], qcd_color,
                       "Anti-iso data $-$ MC $\\times$ FF (QCD est.)")

        ax.set_xlabel("")
        ax.set_ylabel("Events")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=17)
        ax.tick_params(labelbottom=False)
        cms_label(ax, cfg)

        # ---- ratio panel: the effective per-bin fake factor
        raw, raw_unc = curves[RAW]
        ffd, ff_unc = curves[FF]
        safe_raw = np.where(raw > 0, raw, np.nan)
        ratio = ffd / safe_raw
        # same events in both fills -> errors are correlated; the raw-count
        # relative error is the honest per-bin uncertainty of the average FF
        ratio_unc = ratio * np.where(raw > 0, raw_unc / safe_raw, 0.0)

        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt="o",
                     color=qcd_color, markersize=5)
        closure = cfg.qcd.ff_weight and cfg.column_of(var) == cfg.qcd.ff_weight.strip()
        if closure:
            rax.plot(edges, edges, linestyle="--", color="gray", linewidth=1)
        # robust upper limit: bins with raw ~ 0 send single ratios sky-high
        finite = np.isfinite(ratio)
        top = 1.4 * float(np.percentile(ratio[finite], 90)) if finite.any() else 0.0
        if closure:
            top = max(top, 1.1 * float(edges[-1]))
        rax.set_ylim(0, top or 1.0)
        rax.set_ylabel("weighted / raw")
        rax.set_xlabel(var_label(cfg, var))

        save(fig, outdir / "ffcheck", var, console)
