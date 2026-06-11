"""Fit result rendering: prefit/postfit stacks and the NLL scan."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.combine import dc_name, fitdiag_file, scan_file
from wham.config import AnalysisConfig
from wham.fitconfig import FitConfig
from wham.render.common import cms_label, draw_unc_band, save, var_label


def _tag(fit: FitConfig) -> str:
    parts = []
    if fit.asimov.enabled:
        parts.append("Asimov")
    if fit.toy.asymmetry != 0.0:
        parts.append(f"TOY A={fit.toy.asymmetry:g}")
    return " | ".join(parts)


def _stamp(ax, fit: FitConfig) -> None:
    tag = _tag(fit)
    if tag:
        ax.text(0.97, 0.97, tag, transform=ax.transAxes, ha="right", va="top",
                fontsize=13, color="crimson", fontweight="bold")


def _color_map(fit: FitConfig, cfg: AnalysisConfig) -> dict[str, tuple[str, str]]:
    """datacard name -> (display label, color)."""
    out = {}
    for proc, pcfg in cfg.processes.items():
        out[dc_name(proc)] = (proc, pcfg.color)
    sig_color = cfg.processes[fit.signal].color
    out[dc_name(f"{fit.signal}_cpeven")] = (f"{fit.signal} CP-even", sig_color)
    out[dc_name(f"{fit.signal}_cpodd")] = (f"{fit.signal} CP-odd", "tab:pink")
    return out


def _render_shape_dir(fit: FitConfig, cfg: AnalysisConfig, shapes, title: str,
                      outdir: Path, fname: str, console) -> None:
    colors = _color_map(fit, cfg)
    edges = None
    stack_vals, stack_labels, stack_colors = [], [], []

    for key in shapes.keys(cycle=False):
        name = key.split("/")[-1]
        if name in ("data", "total", "total_background", "total_signal", "total_covar"):
            continue
        th1 = shapes[key]
        if edges is None:
            edges = th1.axis().edges()
        label, color = colors.get(name, (name, "tab:gray"))
        stack_vals.append(th1.values())
        stack_labels.append(label)
        stack_colors.append(color)

    if edges is None:
        if console is not None:
            console.print(f"  [yellow]no shapes found for {title} — skipped[/yellow]")
        return

    total = shapes["total"]
    total_vals = total.values()
    total_unc = np.sqrt(np.maximum(total.variances(), 0.0))
    data = shapes["data"]  # TGraphAsymmErrors
    data_y = data.values(axis="y")
    centers = 0.5 * (edges[:-1] + edges[1:])
    data_unc = np.sqrt(np.maximum(data_y, 0.0))

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    order = np.argsort([v.sum() for v in stack_vals])
    hep.histplot([stack_vals[i] for i in order], bins=edges, stack=True, histtype="fill",
                 color=[stack_colors[i] for i in order],
                 label=[stack_labels[i] for i in order],
                 edgecolor="black", linewidth=0.5, ax=ax)
    draw_unc_band(ax, edges, np.maximum(total_vals - total_unc, 0), total_vals + total_unc,
                  label="Unc.")
    ax.errorbar(centers, data_y, yerr=data_unc, fmt="o", color="black",
                markersize=5, label="Data", zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("Events")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12, ncol=2)
    ax.tick_params(labelbottom=False)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold")
    cms_label(ax, cfg.lumi)
    _stamp(ax, fit)

    safe = np.where(total_vals > 0, total_vals, np.nan)
    rax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    draw_unc_band(rax, edges, 1 - np.where(total_vals > 0, total_unc / safe, 0),
                  1 + np.where(total_vals > 0, total_unc / safe, 0))
    rax.errorbar(centers, data_y / safe, yerr=data_unc / safe, fmt="o",
                 color="black", markersize=5)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("Data / Fit")
    rax.set_xlabel(var_label(cfg, fit.variable))

    save(fig, outdir, fname, console)


def render_fit(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path, console=None) -> None:
    import uproot

    from wham.render.common import set_style

    set_style()

    # ---- prefit / postfit stacks
    fd_path = fitdir / fitdiag_file(fit)
    if fd_path.is_file():
        with uproot.open(fd_path) as fd:
            for dirname, label, fname in (
                ("shapes_prefit", "Prefit", "prefit"),
                ("shapes_fit_s", "Postfit (s+b)", "postfit"),
            ):
                key = f"{dirname}/{fit.bin}"
                if key in fd:
                    _render_shape_dir(fit, cfg, fd[key], label, fitdir, fname, console)

    # ---- NLL scan
    scan_path = fitdir / scan_file(fit)
    if scan_path.is_file():
        with uproot.open(scan_path) as f:
            tree = f["limit"]
            poi = tree[fit.poi()].array(library="np")
            dnll = tree["deltaNLL"].array(library="np")

        grid = dnll > 0  # first entry is the best fit (deltaNLL == 0)
        order = np.argsort(poi[grid])
        x, y = poi[grid][order], 2 * dnll[grid][order]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(x, y, "-o", color="tab:blue", markersize=4)
        for level, label in ((1.0, r"68% CL"), (3.84, r"95% CL")):
            ax.axhline(level, linestyle="--", color="gray", linewidth=1)
            ax.text(x[-1], level * 1.02, label, ha="right", fontsize=12, color="gray")
        best = float(poi[~grid][0]) if np.any(~grid) else float(x[np.argmin(y)])
        ax.axvline(best, linestyle=":", color="tab:red", linewidth=1)

        xlabel = r"$\alpha$ [rad]" if fit.poi() == "alpha" else r"$r$"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$-2\,\Delta\,\mathrm{ln}\,L$")
        ax.set_ylim(bottom=0)
        ax.text(0.03, 0.97, f"best fit {fit.poi()} = {best:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=14)
        cms_label(ax, cfg.lumi)
        _stamp(ax, fit)
        save(fig, fitdir, "nll_scan", console)
