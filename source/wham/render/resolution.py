"""Resolution plots: stacked per-process distribution of the derived variable."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.config import AnalysisConfig
from wham.fill import REGION_NOMINAL, resolution_name
from wham.render.common import cms_label, save, stack_components, var_label


def render_resolution(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    pair_by_name = {resolution_name(reco, ref): (reco, ref) for reco, ref in cfg.plots.resolution}

    for (family, name), h in sorted(hists.items()):
        if family != "resolution":
            continue
        pair = pair_by_name.get(name)
        if pair is None:
            continue
        reco, ref = pair
        if only_vars and reco not in only_vars and ref not in only_vars:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        stack, labels, colors = stack_components(cfg, h, REGION_NOMINAL)
        if stack:
            hep.histplot(stack, stack=True, histtype="fill", color=colors,
                         label=labels, edgecolor="black", linewidth=0.5, ax=ax)

            total = np.sum([s.view()["value"] for s in stack], axis=0)
            centers = h.axes[-1].centers
            n = total.sum()
            if n > 0:
                mean = float((centers * total).sum() / n)
                rms = float(np.sqrt(np.maximum((centers**2 * total).sum() / n - mean**2, 0)))
                ax.plot([], [], " ", label=f"mean = {mean:.3f}\nRMS = {rms:.3f}")

        is_angle = cfg.variables[ref].kind == "angle"
        xlabel = (
            f"$\\Delta$({var_label(cfg, reco)}, {var_label(cfg, ref)})"
            if is_angle
            else f"({reco} $-$ {ref}) / {ref}"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Events")
        ax.legend(fontsize=13)
        ax.set_ylim(bottom=0)
        cms_label(ax, cfg.lumi)
        save(fig, outdir / "resolution", name, console)
