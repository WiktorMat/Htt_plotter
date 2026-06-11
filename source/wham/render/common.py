"""Shared CMS-style plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

from wham.config import AnalysisConfig  # noqa: E402

_style_set = False


def set_style() -> None:
    global _style_set
    if not _style_set:
        hep.style.use("CMS")
        _style_set = True


def cms_label(ax, lumi: float) -> None:
    hep.cms.label("Private Work", data=True, lumi=round(lumi / 1000.0, 1), com=13.6, ax=ax)


def save(fig, outdir: Path, name: str, console=None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if console is not None:
        console.print(f"  [green]saved[/green] {outdir / name}.png")


def var_label(cfg: AnalysisConfig, var: str) -> str:
    vcfg = cfg.variables.get(var)
    return vcfg.label if (vcfg is not None and vcfg.label) else var


def slice_1d(h: Any, process: str, region: str):
    return h[{"process": process, "region": region, "variation": "nominal"}]


def stack_components(
    cfg: AnalysisConfig, h: Any, region: str
) -> tuple[list[Any], list[str], list[str]]:
    """Non-empty MC/QCD 1D hists in draw order with labels and colors."""
    hists, labels, colors = [], [], []
    for proc in cfg.stack_order():
        if proc not in list(h.axes["process"]):
            continue
        h1 = slice_1d(h, proc, region)
        if not np.any(h1.view()["value"]):
            continue
        hists.append(h1)
        labels.append(proc)
        colors.append(cfg.processes[proc].color)
    return hists, labels, colors


def total_mc(h: Any, cfg: AnalysisConfig, region: str) -> tuple[np.ndarray, np.ndarray]:
    """(counts, sumw2) summed over MC + QCD processes."""
    nbins = h.axes[-1].size
    counts = np.zeros(nbins)
    sumw2 = np.zeros(nbins)
    for proc in cfg.stack_order():
        if proc not in list(h.axes["process"]):
            continue
        view = slice_1d(h, proc, region).view()
        counts += view["value"]
        sumw2 += view["variance"]
    return counts, sumw2


def draw_unc_band(ax, edges: np.ndarray, low: np.ndarray, high: np.ndarray, label=None) -> None:
    ax.fill_between(
        edges,
        np.r_[low, low[-1]],
        np.r_[high, high[-1]],
        step="post",
        facecolor="none",
        edgecolor="black",
        hatch="////",
        linewidth=0.0,
        alpha=0.5,
        label=label,
    )
