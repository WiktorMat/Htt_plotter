"""Stacked control plots: weighted MC stack + data points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.config import AnalysisConfig
from wham.fill import REGION_NOMINAL
from wham.render.common import (
    cms_label,
    save,
    slice_1d,
    stack_components,
    var_label,
)


def render_control(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    for (family, var), h in sorted(hists.items()):
        if family != "control" or (only_vars and var not in only_vars):
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        stack, labels, colors = stack_components(cfg, h, REGION_NOMINAL)
        if stack:
            hep.histplot(stack, stack=True, histtype="fill", color=colors,
                         label=labels, edgecolor="black", linewidth=0.5, ax=ax)

        data_proc = cfg.data_process()
        if data_proc is not None:
            h_data = slice_1d(h, data_proc, REGION_NOMINAL)
            if np.any(h_data.view()["value"]):
                hep.histplot(h_data, histtype="errorbar", color="black",
                             label="Data", yerr=True, ax=ax)

        ax.set_xlabel(var_label(cfg, var))
        ax.set_ylabel("Events")
        ax.legend(fontsize=14)
        ax.set_ylim(bottom=0)
        cms_label(ax, cfg.lumi)
        save(fig, outdir / "control", var, console)
