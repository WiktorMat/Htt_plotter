"""CP plots: even (SM) vs odd (PS) weighted distributions + integrated asymmetry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.config import AnalysisConfig
from wham.render.common import cms_label, save, var_label


def render_cp(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    for (family, var), h in sorted(hists.items()):
        if family != "cp" or (only_vars and var not in only_vars):
            continue

        nbins = h.axes[-1].size
        even = np.zeros(nbins)
        odd = np.zeros(nbins)
        for proc, pcfg in cfg.processes.items():
            if pcfg.kind != "mc" or proc not in list(h.axes["process"]):
                continue
            even += h[{"process": proc, "region": "even", "variation": "nominal"}].view()["value"]
            odd += h[{"process": proc, "region": "odd", "variation": "nominal"}].view()["value"]

        if not (np.any(even) or np.any(odd)):
            if console is not None:
                console.print(f"  [yellow]cp/{var}: empty (no wt_cp columns?) — skipped[/yellow]")
            continue

        edges = h.axes[-1].edges
        fig, ax = plt.subplots(figsize=(10, 8))
        hep.histplot([even, odd], bins=edges, histtype="step",
                     color=["tab:blue", "tab:red"],
                     label=["CP-even (SM)", "CP-odd (PS)"], linewidth=2, ax=ax)

        denom = even.sum() + odd.sum()
        if denom > 0:
            asym = (even.sum() - odd.sum()) / denom
            ax.plot([], [], " ", label=f"A = {asym:.4f}")

        ax.set_xlabel(var_label(cfg, var))
        ax.set_ylabel("Weighted events")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=13)
        cms_label(ax, cfg.lumi)
        save(fig, outdir / "cp", var, console)
