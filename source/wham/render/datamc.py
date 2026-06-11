"""Data/MC ratio plots: MC+QCD stack, stat band, data points, ratio panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.config import AnalysisConfig
from wham.render.common import (
    cms_label,
    draw_unc_band,
    save,
    slice_1d,
    stack_components,
    total_mc,
    var_label,
)


def signal_region(qcd_method: str) -> str:
    return "OS" if qcd_method == "ss" else "OS_iso"


def render_datamc(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    region = signal_region(cfg.qcd.method)
    data_proc = cfg.data_process()

    for (family, var), h in sorted(hists.items()):
        if family != "datamc" or (only_vars and var not in only_vars):
            continue
        if data_proc is None or region not in list(h.axes["region"]):
            continue

        edges = h.axes[-1].edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        data_view = slice_1d(h, data_proc, region).view()
        data = data_view["value"]
        data_unc = np.sqrt(np.maximum(data_view["variance"], 0.0))

        mc, mc_sumw2 = total_mc(h, cfg, region)
        mc_unc = np.sqrt(np.maximum(mc_sumw2, 0.0))

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        rax = fig.add_subplot(gs[1], sharex=ax)

        stack, labels, colors = stack_components(cfg, h, region)
        if stack:
            hep.histplot(stack, stack=True, histtype="fill", color=colors,
                         label=labels, edgecolor="black", linewidth=0.5, ax=ax)

        draw_unc_band(ax, edges, np.maximum(mc - mc_unc, 0.0), mc + mc_unc,
                      label="Stat. unc.")
        ax.errorbar(centers, data, yerr=data_unc, fmt="o", color="black",
                    markersize=5, label="Data", zorder=5)

        ax.set_xlabel("")  # mplhep copies the hist axis name; the ratio panel owns it
        ax.set_ylabel("Events")
        ax.set_ylim(bottom=0)
        # legend reads top-of-stack first: Data, then MC top to bottom, band last
        handles, names = ax.get_legend_handles_labels()
        by_label = dict(zip(names, handles))
        order = ["Data", *reversed(labels), "Stat. unc."]
        ax.legend([by_label[n] for n in order if n in by_label],
                  [n for n in order if n in by_label], fontsize=17)
        ax.tick_params(labelbottom=False)
        cms_label(ax, cfg)

        # ---- ratio panel
        safe_mc = np.where(mc > 0, mc, np.nan)
        ratio = data / safe_mc
        ratio_unc = data_unc / safe_mc
        rel_mc = np.where(mc > 0, mc_unc / safe_mc, 0.0)

        rax.axhline(1.0, linestyle="--", color="black", linewidth=1)
        draw_unc_band(rax, edges, 1.0 - rel_mc, 1.0 + rel_mc)
        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt="o", color="black", markersize=5)
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("Data / MC")
        rax.set_xlabel(var_label(cfg, var))

        save(fig, outdir / "datamc", var, console)
