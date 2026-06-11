"""Fit result rendering: per-category prefit/postfit stacks, pulls+impacts,
and the 1D/2D NLL scans."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wham.combine import (
    FITRESULT_FILE,
    SHAPES_FILE,
    fitdiag_file,
    scan_file,
    scan_tag,
)
from wham.config import AnalysisConfig
from wham.fitconfig import FitConfig, ScanCfg, dc_name
from wham.render.common import cms_label, cms_label_split, draw_unc_band, save, var_label

# colors for the 2nd, 3rd, ... component template of a process (the first
# keeps the process color from the analysis config)
_COMPONENT_COLORS = ("tab:pink", "tab:cyan", "tab:gray", "tab:olive")

_GREEK = {"alpha", "beta", "gamma", "delta", "mu", "phi", "theta", "tau", "kappa"}


def poi_label(poi: str) -> str:
    return rf"$\{poi}$" if poi in _GREEK else poi


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
    """datacard template name -> (display label, color)."""
    out = {}
    for proc, pcfg in cfg.processes.items():
        out[dc_name(proc)] = (pcfg.label or proc, pcfg.color)
    for proc, pm in fit.model.processes.items():
        if pm.components is None:
            continue
        base_label = cfg.processes[proc].label or proc
        for i, comp in enumerate(pm.components):
            color = cfg.processes[proc].color if i == 0 else _COMPONENT_COLORS[
                (i - 1) % len(_COMPONENT_COLORS)]
            out[dc_name(f"{proc}_{comp}")] = (f"{base_label} ({comp})", color)
    return out


def _stack_dc_order(fit: FitConfig, cfg: AnalysisConfig) -> list[str]:
    """Datacard template names in the configured stack draw order."""
    order = []
    for proc in cfg.stack_order():
        pm = fit.model.processes.get(proc)
        if pm is not None and pm.components is not None:
            order += [dc_name(f"{proc}_{comp}") for comp in pm.components]
        else:
            order.append(dc_name(proc))
    return order


def _render_shape_dir(fit: FitConfig, cfg: AnalysisConfig, shapes, title: str,
                      outdir: Path, fname: str, console, edges=None,
                      xlabel: str = "") -> None:
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
    # combine pads every channel's saved shapes to the largest channel's bin
    # count; the true bin count comes from our exported edges
    nbins = len(edges) - 1

    stack_vals, stack_labels, stack_colors = [], [], []
    ordered = [n for n in _stack_dc_order(fit, cfg) if n in available]
    ordered += sorted(set(available) - set(ordered))  # anything unexpected on top
    for name in ordered:
        label, color = colors.get(name, (name, "tab:gray"))
        stack_vals.append(available[name].values()[:nbins])
        stack_labels.append(label)
        stack_colors.append(color)

    total = shapes["total"]
    total_vals = total.values()[:nbins]
    total_unc = np.sqrt(np.maximum(total.variances()[:nbins], 0.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    # the data graph has one point per populated bin, at x = bin index + 0.5
    data = shapes["data"]  # TGraphAsymmErrors
    gx = data.values(axis="x")
    gy = data.values(axis="y")
    data_y = np.zeros(nbins)
    idx = np.floor(gx).astype(int)
    keep = (idx >= 0) & (idx < nbins)
    data_y[idx[keep]] = gy[keep]
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
    rax.set_xlabel(xlabel)

    save(fig, outdir, fname, console)


# ------------------------------------------------------------- pulls/impacts


def _render_pulls(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path, console) -> None:
    path = fitdir / FITRESULT_FILE
    if not path.is_file():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    params = result.get("params", {})
    impacts = result.get("impacts", {})

    pois = [p for p in fit.model.pois if p in params]
    rate_params = [s.name for s in fit.systematics
                   if s.effect == "rateParam" and s.name in params]
    free = pois + rate_params
    constrained = [n for n in params
                   if n not in free and not n.startswith("prop_bin")]

    def importance(name: str) -> float:
        vals = [abs(impacts.get(p, {}).get(name, 0.0)) for p in pois]
        return max(vals) if vals else abs(params[name]["value"])

    # rateParams get impact bars but no pull (they are unconstrained)
    rows = sorted(constrained + rate_params, key=importance)  # largest on top
    if not rows and not free:
        return

    n_imp = len(pois)
    head = 0.75 * len(free) + 0.8  # data-units of headroom for the summary
    fig_h = max(4.5, 1.8 + 0.55 * len(rows) + 0.45 * len(free))
    fig = plt.figure(figsize=(11, fig_h))
    gs = gridspec.GridSpec(1, 1 + n_imp, width_ratios=[2.2] + [1.0] * n_imp,
                           wspace=0.07)
    axp = fig.add_subplot(gs[0])
    iaxes = [fig.add_subplot(gs[i + 1], sharey=axp) for i in range(n_imp)]

    y = np.arange(len(rows))
    axp.axvspan(-2, 2, color="gold", alpha=0.35, zorder=0)
    axp.axvspan(-1, 1, color="yellowgreen", alpha=0.45, zorder=1)
    axp.axvline(0, color="gray", linewidth=1, zorder=2)
    for yy in y[:-1]:
        axp.axhline(yy + 0.5, color="gray", linewidth=0.4, alpha=0.4, zorder=0)

    pull_rows = [(i, params[n]) for i, n in enumerate(rows) if n not in rate_params]
    if pull_rows:
        axp.errorbar([p["value"] for _, p in pull_rows], [i for i, _ in pull_rows],
                     xerr=[p["error"] for _, p in pull_rows],
                     fmt="o", color="black", markersize=6, zorder=5)
    lim = 2.5
    if pull_rows:
        lim = max(lim, max(abs(p["value"]) + p["error"] for _, p in pull_rows) + 0.4)
    axp.set_xlim(-lim, lim)
    axp.set_yticks(y, rows, fontsize=13)
    axp.set_ylim(-0.6, max(len(rows) - 0.4, 0.4) + head)
    axp.set_xlabel(r"$(\hat{\theta} - \theta_0) / \Delta\theta$", fontsize=15)
    axp.tick_params(labelsize=13)

    # free-parameter summary (POIs + rateParams), top left
    if free:
        lines = []
        for name in free:
            p = params[name]
            label = name.replace("_", r"\_")
            lines.append(rf"${label} = {p['value']:.3f} \pm {p['error']:.3f}$")
        axp.text(0.03, 0.985, "\n".join(lines), transform=axp.transAxes,
                 ha="left", va="top", fontsize=14)

    for iax, poi in zip(iaxes, pois):
        vals = np.array([impacts.get(poi, {}).get(n, 0.0) for n in rows])
        iax.barh(y, vals, height=0.55, color="tab:blue", alpha=0.8)
        iax.axvline(0, color="gray", linewidth=1)
        for yy in y[:-1]:
            iax.axhline(yy + 0.5, color="gray", linewidth=0.4, alpha=0.4, zorder=0)
        span = float(np.max(np.abs(vals))) if len(vals) and np.max(np.abs(vals)) > 0 else 1.0
        iax.set_xlim(-1.4 * span, 1.4 * span)
        iax.set_ylim(*axp.get_ylim())
        iax.set_xlabel(rf"$\Delta$ {poi_label(poi)}", fontsize=15)
        iax.tick_params(labelleft=False, labelsize=11)
        iax.xaxis.get_offset_text().set_fontsize(10)

    cms_label_split(axp, iaxes[-1] if iaxes else axp, cfg)
    _stamp(iaxes[0] if iaxes else axp, fit)
    save(fig, fitdir, "pulls", console)


# ------------------------------------------------------------- scans


def _scan_arrays(fitdir: Path, fit: FitConfig, scan: ScanCfg, branches: list[str]):
    import uproot

    path = fitdir / scan_file(fit, scan)
    if not path.is_file():
        return None
    with uproot.open(path) as f:
        tree = f["limit"]
        return {b: tree[b].array(library="np") for b in [*branches, "deltaNLL"]}


def _render_scan_1d(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path,
                    scan: ScanCfg, console) -> None:
    poi = scan.pois[0]
    arrays = _scan_arrays(fitdir, fit, scan, [poi])
    if arrays is None:
        return
    vals, dnll = arrays[poi], arrays["deltaNLL"]

    grid = dnll > 0  # first entry is the best fit (deltaNLL == 0)
    order = np.argsort(vals[grid])
    x, y = vals[grid][order], 2 * dnll[grid][order]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, "-o", color="tab:blue", markersize=4)
    for level, label in ((1.0, r"68% CL"), (3.84, r"95% CL")):
        ax.axhline(level, linestyle="--", color="gray", linewidth=1)
        ax.text(x[-1], level * 1.02, label, ha="right", fontsize=12, color="gray")
    best = float(vals[~grid][0]) if np.any(~grid) else float(x[np.argmin(y)])
    ax.axvline(best, linestyle=":", color="tab:red", linewidth=1)

    ax.set_xlabel(poi_label(poi))
    ax.set_ylabel(r"$-2\,\Delta\,\mathrm{ln}\,L$")
    ax.set_ylim(bottom=0)
    ax.text(0.03, 0.97, f"best fit {poi} = {best:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=14)
    cms_label(ax, cfg)
    _stamp(ax, fit)
    save(fig, fitdir, f"nll_scan_{scan_tag(scan)}", console)


def _render_scan_2d(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path,
                    scan: ScanCfg, console) -> None:
    p1, p2 = scan.pois
    arrays = _scan_arrays(fitdir, fit, scan, [p1, p2])
    if arrays is None:
        return
    x, y, dnll = arrays[p1], arrays[p2], arrays["deltaNLL"]

    grid = dnll > 0
    finite = grid & np.isfinite(dnll)
    if finite.sum() < 4:
        if console is not None:
            console.print(f"  [yellow]2D scan {scan_tag(scan)}: too few points — skipped[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(10, 9))
    # 68% / 95% CL for 2 parameters: 2*deltaNLL = 2.30 / 5.99
    ax.tricontour(x[finite], y[finite], 2 * dnll[finite],
                  levels=[2.30, 5.99], colors=["tab:blue", "tab:red"],
                  linewidths=2)
    handles = [
        mlines.Line2D([], [], color="tab:blue", linewidth=2, label="68% CL"),
        mlines.Line2D([], [], color="tab:red", linewidth=2, label="95% CL"),
    ]
    if np.any(~grid):
        bx, by = float(x[~grid][0]), float(y[~grid][0])
        ax.plot(bx, by, "*", color="black", markersize=16, zorder=5)
        handles.append(mlines.Line2D([], [], color="black", marker="*", linestyle="",
                                     markersize=12, label="Best fit"))

    ax.set_xlabel(poi_label(p1))
    ax.set_ylabel(poi_label(p2))
    ax.legend(handles=handles, fontsize=14, loc="upper left")
    cms_label(ax, cfg)
    _stamp(ax, fit)
    save(fig, fitdir, f"nll_scan_{scan_tag(scan)}", console)


# ------------------------------------------------------------- driver


def render_fit(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path, console=None) -> None:
    import uproot

    from wham.render.common import set_style

    set_style()
    multi = len(fit.categories) > 1

    # combine saves shapes with unit-width bins; recover the real edges per
    # category from the exported shapes.root
    edges: dict[str, np.ndarray] = {}
    shapes_path = fitdir / SHAPES_FILE
    if shapes_path.is_file():
        with uproot.open(shapes_path) as sf:
            for cat in fit.categories:
                key = f"{cat.name}/data_obs"
                if key in sf:
                    edges[cat.name] = sf[key].axis().edges()

    # ---- prefit / postfit stacks, one per category
    fd_path = fitdir / fitdiag_file(fit)
    if fd_path.is_file():
        with uproot.open(fd_path) as fd:
            for cat in fit.categories:
                for dirname, label, fname in (
                    ("shapes_prefit", "Prefit", "prefit"),
                    ("shapes_fit_s", "Postfit (s+b)", "postfit"),
                ):
                    key = f"{dirname}/{cat.name}"
                    if key not in fd:
                        continue
                    _render_shape_dir(
                        fit, cfg, fd[key],
                        f"{label} — {cat.name}" if multi else label,
                        fitdir, f"{fname}_{cat.name}" if multi else fname,
                        console, edges=edges.get(cat.name),
                        xlabel=var_label(cfg, cat.variable),
                    )

    # ---- nuisance pulls + POI impacts
    _render_pulls(fit, cfg, fitdir, console)

    # ---- NLL scans
    for scan in fit.resolved_scans():
        if len(scan.pois) == 1:
            _render_scan_1d(fit, cfg, fitdir, scan, console)
        else:
            _render_scan_2d(fit, cfg, fitdir, scan, console)
