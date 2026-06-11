"""Fit result rendering: prefit/postfit stacks and the NLL scan."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.combine import FITRESULT_FILE, SHAPES_FILE, dc_name, fitdiag_file, scan_file
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
        out[dc_name(proc)] = (pcfg.label or proc, pcfg.color)
    sig = cfg.processes[fit.signal]
    sig_label = sig.label or fit.signal
    out[dc_name(f"{fit.signal}_cpeven")] = (f"{sig_label} CP-even", sig.color)
    out[dc_name(f"{fit.signal}_cpodd")] = (f"{sig_label} CP-odd", "tab:pink")
    return out


def _stack_dc_order(fit: FitConfig, cfg: AnalysisConfig) -> list[str]:
    """Datacard template names in the configured stack draw order."""
    order = []
    for proc in cfg.stack_order():
        if fit.mode == "cp" and proc == fit.signal:
            order += [dc_name(f"{proc}_cpeven"), dc_name(f"{proc}_cpodd")]
        else:
            order.append(dc_name(proc))
    return order


def _render_shape_dir(fit: FitConfig, cfg: AnalysisConfig, shapes, title: str,
                      outdir: Path, fname: str, console, edges=None) -> None:
    colors = _color_map(fit, cfg)

    # combine's saved shapes use unit-width bins; the real edges come from
    # our shapes.root (the `edges` argument), these are only the fallback
    available: dict = {}
    for key in shapes.keys(cycle=False):
        name = key.split("/")[-1]
        if name not in ("data", "total", "total_background", "total_signal", "total_covar"):
            available[name] = shapes[key]

    if not available:
        if console is not None:
            console.print(f"  [yellow]no shapes found for {title} — skipped[/yellow]")
        return
    if edges is None:
        edges = next(iter(available.values())).axis().edges()

    stack_vals, stack_labels, stack_colors = [], [], []
    ordered = [n for n in _stack_dc_order(fit, cfg) if n in available]
    ordered += sorted(set(available) - set(ordered))  # anything unexpected on top
    for name in ordered:
        label, color = colors.get(name, (name, "tab:gray"))
        stack_vals.append(available[name].values())
        stack_labels.append(label)
        stack_colors.append(color)

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

    hep.histplot(stack_vals, bins=edges, stack=True, histtype="fill",
                 color=stack_colors, label=stack_labels,
                 edgecolor="black", linewidth=0.5, ax=ax)
    draw_unc_band(ax, edges, np.maximum(total_vals - total_unc, 0), total_vals + total_unc,
                  label="Unc.")
    ax.errorbar(centers, data_y, yerr=data_unc, fmt="o", color="black",
                markersize=5, label="Data", zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("Events")
    ax.set_ylim(bottom=0)
    # legend reads top-of-stack first: Data, then MC top to bottom, band last
    handles, names = ax.get_legend_handles_labels()
    by_label = dict(zip(names, handles))
    legend_order = ["Data", *reversed(stack_labels), "Unc."]
    ax.legend([by_label[n] for n in legend_order if n in by_label],
              [n for n in legend_order if n in by_label], fontsize=13)
    ax.tick_params(labelbottom=False)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold")
    cms_label(ax, cfg)
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


def _render_pulls(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path, console) -> None:
    import json

    path = fitdir / FITRESULT_FILE
    if not path.is_file():
        return
    params = json.loads(path.read_text(encoding="utf-8"))

    # free parameters (POI + rateParams) are reported as values, not pulls;
    # prop_bin* (autoMCStats) would swamp the plot
    free_names = [fit.poi()] + [s.name for s in fit.systematics if s.effect == "rateParam"]
    free = [(n, params[n]) for n in free_names if n in params]
    pulls = sorted(
        (n, p) for n, p in params.items()
        if n not in free_names and not n.startswith("prop_bin")
    )
    if not pulls and not free:
        return

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.6 * len(pulls) + 3.0)))
    ax.axvspan(-2, 2, color="gold", alpha=0.35, zorder=0)
    ax.axvspan(-1, 1, color="yellowgreen", alpha=0.45, zorder=1)
    ax.axvline(0, color="gray", linewidth=1, zorder=2)

    y = np.arange(len(pulls))
    if pulls:
        values = np.array([p["value"] for _, p in pulls])
        errors = np.array([p["error"] for _, p in pulls])
        ax.errorbar(values, y, xerr=errors, fmt="o", color="black",
                    markersize=6, zorder=5)
        ax.set_xlim(-max(2.5, float(np.max(np.abs(values) + errors)) + 0.5),
                    max(2.5, float(np.max(np.abs(values) + errors)) + 0.5))
    else:
        ax.set_xlim(-2.5, 2.5)
    ax.set_yticks(y, [n for n, _ in pulls])
    # headroom above the nuisances for the free-parameter summary
    ax.set_ylim(-0.7, len(pulls) + max(1.5, 0.8 * len(free)))
    ax.set_xlabel(r"$(\hat{\theta} - \theta_0) / \Delta\theta$")

    if free:
        labels = {n: n.replace("_", r"\_") for n, _ in free}
        text = "\n".join(
            rf"${labels[n]} = {p['value']:.3f} \pm {p['error']:.3f}$" for n, p in free
        )
        ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=15)

    cms_label(ax, cfg)
    _stamp(ax, fit)
    save(fig, fitdir, "pulls", console)


def render_fit(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path, console=None) -> None:
    import uproot

    from wham.render.common import set_style

    set_style()

    # ---- prefit / postfit stacks
    # combine saves shapes with unit-width bins; recover the variable's real
    # edges from the exported shapes.root
    edges = None
    shapes_path = fitdir / SHAPES_FILE
    if shapes_path.is_file():
        with uproot.open(shapes_path) as sf:
            edges = sf[f"{fit.bin}/data_obs"].axis().edges()

    fd_path = fitdir / fitdiag_file(fit)
    if fd_path.is_file():
        with uproot.open(fd_path) as fd:
            for dirname, label, fname in (
                ("shapes_prefit", "Prefit", "prefit"),
                ("shapes_fit_s", "Postfit (s+b)", "postfit"),
            ):
                key = f"{dirname}/{fit.bin}"
                if key in fd:
                    _render_shape_dir(fit, cfg, fd[key], label, fitdir, fname,
                                      console, edges=edges)

    # ---- nuisance pulls / free-parameter summary
    _render_pulls(fit, cfg, fitdir, console)

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
        cms_label(ax, cfg)
        _stamp(ax, fit)
        save(fig, fitdir, "nll_scan", console)
