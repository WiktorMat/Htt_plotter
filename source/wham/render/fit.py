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
from wham.fitconfig import FitConfig, ScanCfg, all_pois, dc_name

# beyond this many POIs the per-POI impact columns become unreadable; show
# impacts for a headline subset (morph POIs first) and rely on the POI summary
# forest plot + fitresult.json for the rest
MAX_IMPACT_COLS = 6
from wham.render.common import (
    cms_label,
    cms_label_split,
    draw_unc_band,
    draw_unroll_guides,
    save,
    var_label,
)

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
                      xlabel: str = "", var: str | None = None) -> None:
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
    # the data graph has one point per populated bin. Standalone combine puts it
    # on the CMS_th1x index axis (x = bin index + 0.5); CombineHarvester keeps the
    # physical observable axis (x = m_vis bin centre) — handle both.
    data = shapes["data"]  # TGraphAsymmErrors
    gx = data.values(axis="x")
    gy = data.values(axis="y")
    data_y = np.zeros(nbins)
    if len(gx):
        if float(np.max(gx)) <= nbins + 1.0:          # bin-index axis
            idx = np.floor(gx).astype(int)
        else:                                          # physical observable axis
            idx = np.searchsorted(edges, gx, side="right") - 1
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
    if var is not None:
        draw_unroll_guides(ax, cfg, var)
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

    pois = [p for p in all_pois(fit) if p in params]
    rate_params = [s.name for s in fit.systematics
                   if s.effect == "rateParam" and s.name in params]
    free = pois + rate_params
    constrained = [n for n in params
                   if n not in free and not n.startswith("prop_bin")]

    # impact columns: cap to keep the figure readable; show the headline POIs
    # (morph/TES first, then rate). The full set lives in the POI summary plot.
    morph_pois = [p for p in fit.model.morphs if p in params]
    rate_pois = [p for p in fit.model.pois if p in params]
    impact_pois = (morph_pois + rate_pois)[:MAX_IMPACT_COLS]

    def importance(name: str) -> float:
        vals = [abs(impacts.get(p, {}).get(name, 0.0)) for p in impact_pois]
        return max(vals) if vals else abs(params[name]["value"])

    # rateParams get impact bars but no pull (they are unconstrained)
    rows = sorted(constrained + rate_params, key=importance)  # largest on top
    if not rows and not free:
        return

    # the per-free-param value summary is only legible for a handful; with many
    # POIs (e.g. the SF grid) the POI summary forest plot carries them instead
    show_summary = bool(free) and len(free) <= 8
    n_imp = len(impact_pois)
    head = (0.75 * len(free) + 0.8) if show_summary else 0.8
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
    if show_summary:
        lines = []
        for name in free:
            p = params[name]
            label = name.replace("_", r"\_")
            lines.append(rf"${label} = {p['value']:.3f} \pm {p['error']:.3f}$")
        axp.text(0.03, 0.985, "\n".join(lines), transform=axp.transAxes,
                 ha="left", va="top", fontsize=14)

    for iax, poi in zip(iaxes, impact_pois):
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


def _scan_poi_interval(fitdir: Path, fit: FitConfig, poi: str):
    """Profiled (best, err_dn, err_up) of one POI from its NLL scan — the 1D
    scan when declared, else the first 2D scan containing the POI (profiled
    over the other axis). None when no scan output exists."""
    scans = sorted((s for s in fit.resolved_scans() if poi in s.pois),
                   key=lambda s: len(s.pois))
    for scan in scans:
        arrays = _scan_arrays(fitdir, fit, scan, list(scan.pois))
        if arrays is None:
            continue
        v, d = arrays[poi][1:], arrays["deltaNLL"][1:]  # row 0 = the free fit
        keep = np.isfinite(d)
        v, d = v[keep], 2.0 * (d[keep] - min(float(d[keep].min()), 0.0))
        xs = np.unique(v)
        if len(xs) < 3:
            continue
        prof = np.array([d[v == xv].min() for xv in xs])
        prof -= prof.min()
        bi = int(np.argmin(prof))
        best = float(xs[bi])

        def cross(step: int):
            px, py = best, 0.0
            i = bi + step
            while 0 <= i < len(xs):
                yv = float(prof[i])
                if np.isfinite(yv) and yv >= 1.0:
                    return px if yv == py else px + (1.0 - py) / (yv - py) * (float(xs[i]) - px)
                px, py = float(xs[i]), yv
                i += step
            return None

        lo, hi = cross(-1), cross(+1)
        if lo is None and hi is None:
            continue
        return (best,
                best - lo if lo is not None else np.nan,
                hi - best if hi is not None else np.nan)
    return None


def _render_poi_summary(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path, console) -> None:
    """Forest plot of every POI: the FitDiagnostics value (Hesse errors) and,
    for scanned POIs, the MultiDimFit profile (scan minimum + Δ=1 crossings)
    side by side — the two minimizations must agree; a visible offset flags a
    local-minimum FitDiagnostics. Morph POIs (TES) on top, then rate POIs."""
    path = fitdir / FITRESULT_FILE
    if not path.is_file():
        return
    params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
    morph_pois = [p for p in fit.model.morphs if p in params]
    rate_pois = [p for p in fit.model.pois if p in params]
    order = morph_pois + rate_pois
    if len(order) < 2:
        return  # few POIs: the pulls-plot summary already lists them

    vals = [params[p]["value"] for p in order]
    errs = [params[p]["error"] for p in order]
    y = np.arange(len(order))[::-1]  # first POI at the top

    profiles = {p: iv for p in order if (iv := _scan_poi_interval(fitdir, fit, p))}

    fig_h = max(4.0, 0.34 * len(order) + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1, zorder=0)  # no-correction ref
    off = 0.16 if profiles else 0.0
    ax.errorbar(vals, y + off, xerr=errs, fmt="o", color="black", markersize=5,
                zorder=5, label="FitDiagnostics (Hesse)")
    if profiles:
        py_ = np.array([float(y[order.index(p)]) - off for p in profiles])
        pv = np.array([profiles[p][0] for p in profiles])
        pdn = np.array([[profiles[p][1] for p in profiles],
                        [profiles[p][2] for p in profiles]])
        ax.errorbar(pv, py_, xerr=pdn, fmt="s", color="tab:red", markersize=4.5,
                    zorder=5, label="MultiDimFit profile (scan)")
        ax.legend(fontsize=11, loc="best")
        for p in profiles:  # flag local-minimum FitDiagnostics values
            gap = abs(params[p]["value"] - profiles[p][0])
            width = max(profiles[p][1], profiles[p][2], params[p]["error"])
            if console is not None and np.isfinite(width) and gap > width:
                console.print(
                    f"  [yellow]POI {p}: FitDiagnostics ({params[p]['value']:.4f}) and "
                    f"scan profile ({profiles[p][0]:.4f}) disagree by more than 1σ — "
                    f"likely a local minimum; quote the profile[/yellow]"
                )
    if morph_pois and rate_pois:  # separator between the two POI groups
        ax.axhline(y[len(morph_pois) - 1] - 0.5, color="gray", linewidth=0.6, alpha=0.6)
    ax.set_yticks(y, order, fontsize=10)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("postfit value", fontsize=14)
    cms_label(ax, cfg)
    _stamp(ax, fit)
    save(fig, fitdir, "pois", console)


# ------------------------------------------------------------- scans


def _scan_arrays(fitdir: Path, fit: FitConfig, scan: ScanCfg, branches: list[str]):
    import uproot

    path = fitdir / scan_file(fit, scan)
    if not path.is_file():
        return None
    with uproot.open(path) as f:
        tree = f["limit"]
        return {b: tree[b].array(library="np") for b in [*branches, "deltaNLL"]}


def _fitdiag_poi(fitdir: Path, poi: str) -> tuple[float, float] | None:
    """FitDiagnostics (value, Hesse error) of one POI from fitresult.json.
    Overlaid on the scans: the two minimizations MUST agree — a visible gap
    means FitDiagnostics sits in a local minimum (quote the profile)."""
    path = fitdir / FITRESULT_FILE
    if not path.is_file():
        return None
    try:
        params = json.loads(path.read_text()).get("params", {})
    except json.JSONDecodeError:
        return None
    p = params.get(poi)
    if not p or "value" not in p:
        return None
    return float(p["value"]), float(p.get("error", 0.0))


def _render_scan_1d(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path,
                    scan: ScanCfg, console) -> None:
    poi = scan.pois[0]
    arrays = _scan_arrays(fitdir, fit, scan, [poi])
    if arrays is None:
        return
    vals, dnll = arrays[poi], arrays["deltaNLL"]

    # row 0 is the free best fit; keep every grid point (deltaNLL can dip
    # below 0 when the grid finds a deeper minimum) and re-zero to the min
    grid = np.isfinite(dnll)
    grid[0] = False
    order = np.argsort(vals[grid])
    x = vals[grid][order]
    y = 2 * (dnll[grid][order] - min(float(np.min(dnll[grid])), 0.0))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, "-o", color="tab:blue", markersize=4)
    for level, label in ((1.0, r"68% CL"), (3.84, r"95% CL")):
        ax.axhline(level, linestyle="--", color="gray", linewidth=1)
        ax.text(x[-1], level * 1.02, label, ha="right", fontsize=12, color="gray")
    best = float(vals[0]) if float(np.min(dnll[grid])) >= 0 else float(x[np.argmin(y)])
    ax.axvline(best, linestyle=":", color="tab:red", linewidth=1)

    # consistency check (no visual): FitDiagnostics must sit at the scan
    # minimum; a gap means it converged to a local minimum
    fd = _fitdiag_poi(fitdir, poi)
    if fd is not None and console is not None and x[0] <= fd[0] <= x[-1]:
        gap = float(np.interp(fd[0], x, y))
        if gap > 1.0:
            console.print(
                f"  [yellow]scan {scan_tag(scan)}: FitDiagnostics minimum sits "
                f"{gap:.1f} units of 2ΔlnL above the scan minimum — likely a "
                f"local minimum; quote the profile (see pois plot)[/yellow]"
            )

    ax.set_xlabel(poi_label(poi))
    ax.set_ylabel(r"$-2\,\Delta\,\mathrm{ln}\,L$")
    ax.set_ylim(bottom=0)
    ax.text(0.03, 0.97, f"best fit {poi} = {best:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=14)
    cms_label(ax, cfg)
    _stamp(ax, fit)
    save(fig, fitdir, f"nll_scan_{scan_tag(scan)}", console)


def _poi_axis_label(name: str) -> str:
    """Physics axis title for the common SF POIs; falls back to poi_label."""
    n = name.lower()
    if "tes" in n:
        return r"$\tau_{h}$ energy scale"
    if "tid_sf" in n or "tid" in n:
        return r"$\tau_{h}$ ID scale factor"
    return poi_label(name)


def _profile_curve(axis_vals: np.ndarray, Z: np.ndarray, axis: int):
    """Profiled best fit + asymmetric 1σ of one POI from the plotted surface:
    minimize 2ΔlnL over the other axis, then interpolate the Δ=1 crossings
    either side of the minimum. Returns (best, err_dn, err_up)."""
    prof = np.nanmin(Z, axis=axis)
    prof = prof - np.nanmin(prof)
    bi = int(np.nanargmin(prof))
    best = float(axis_vals[bi])

    def cross(step: int):
        px, py = best, 0.0
        i = bi + step
        while 0 <= i < len(axis_vals):
            v = prof[i]
            if np.isfinite(v) and v >= 1.0:
                return px if v == py else px + (1.0 - py) / (v - py) * (axis_vals[i] - px)
            px, py = float(axis_vals[i]), float(v)
            i += step
        return None

    lo, hi = cross(-1), cross(+1)
    return best, (best - lo if lo is not None else np.nan), (hi - best if hi is not None else np.nan)


def _render_scan_2d(fit: FitConfig, cfg: AnalysisConfig, fitdir: Path,
                    scan: ScanCfg, console) -> None:
    p1, p2 = scan.pois
    arrays = _scan_arrays(fitdir, fit, scan, [p1, p2])
    if arrays is None:
        return
    x, y, dnll = arrays[p1], arrays[p2], arrays["deltaNLL"]

    # row 0 is the free best fit; keep every grid point (deltaNLL can dip
    # below 0 when the grid finds a deeper minimum) and re-zero to the min
    grid = np.isfinite(dnll)
    grid[0] = False
    if grid.sum() < 4:
        if console is not None:
            console.print(f"  [yellow]2D scan {scan_tag(scan)}: too few points — skipped[/yellow]")
        return
    gx, gy = x[grid], y[grid]
    gz = 2.0 * (dnll[grid] - min(float(np.nanmin(dnll[grid])), 0.0))

    # smooth, interpolated -2ΔlnL surface (scipy if available, else raw lattice)
    xs, ys = np.unique(gx), np.unique(gy)
    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
        xi = np.linspace(xs.min(), xs.max(), 240)
        yi = np.linspace(ys.min(), ys.max(), 240)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((gx, gy), gz, (XI, YI), method="cubic")
        nan = ~np.isfinite(ZI)
        if nan.any():
            ZI[nan] = griddata((gx, gy), gz, (XI[nan], YI[nan]), method="linear")
        ZI = gaussian_filter(np.where(np.isfinite(ZI), ZI, np.nanmax(gz)), sigma=1.6)
        ZI -= np.nanmin(ZI)  # smoothing lifts the floor; contours expect min=0
        XG, YG, ZG = xi, yi, ZI
    except Exception:
        z = np.full((len(ys), len(xs)), np.nan)
        z[np.searchsorted(ys, gy), np.searchsorted(xs, gx)] = gz
        XG, YG, ZG = xs, ys, z

    # star, per-POI profiled errors and correlation all come from the SAME
    # surface the contours are drawn from, so they agree by construction
    b1, e1d, e1u = _profile_curve(XG, ZG, axis=0)   # profile out y (rows)
    b2, e2d, e2u = _profile_curve(YG, ZG, axis=1)   # profile out x (cols)
    bx, by = b1, b2
    XF, YF = np.meshgrid(XG, YG)
    inside = np.isfinite(ZG) & (ZG < 2.30)
    corr = 0.0
    if inside.sum() >= 8:  # coordinate moments of the 68% region
        cov = np.cov(XF[inside], YF[inside])
        if cov[0, 0] > 0 and cov[1, 1] > 0:
            corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = float(min(9.0, np.nanmax(ZG)))
    mesh = ax.pcolormesh(XG, YG, ZG, shading="auto", cmap="viridis",
                         vmin=0.0, vmax=vmax, rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label(r"$-2\,\Delta\,\mathrm{ln}\,L$", fontsize=17)
    cbar.ax.tick_params(labelsize=13)

    # 68%/95% CL contours for 2 POIs: -2ΔlnL = 2.30 / 5.99
    cs = ax.contour(XG, YG, ZG, levels=[2.30, 5.99], colors="white",
                    linewidths=[2.4, 1.6], linestyles=["solid", "dashed"], zorder=4)
    ax.clabel(cs, fmt={2.30: "68%", 5.99: "95%"}, fontsize=12, colors="white")

    # best fit + profiled asymmetric error bars
    if np.isfinite(e1d) and np.isfinite(e1u):
        ax.errorbar([bx], [by], xerr=[[e1d], [e1u]], fmt="none",
                    ecolor="red", elinewidth=2, capsize=4, zorder=5)
    if np.isfinite(e2d) and np.isfinite(e2u):
        ax.errorbar([bx], [by], yerr=[[e2d], [e2u]], fmt="none",
                    ecolor="red", elinewidth=2, capsize=4, zorder=5)
    ax.plot(bx, by, "*", color="red", markeredgecolor="black", markersize=16, zorder=6)

    def _fmt(v, ed, eu):  # extra precision for small (e.g. TES) errors
        p = 4 if max(abs(ed), abs(eu)) < 0.01 else 3
        return f"${v:.{p}f}_{{-{ed:.{p}f}}}^{{+{eu:.{p}f}}}$"

    txt = (f"{_poi_axis_label(p1)}: {_fmt(bx, e1d, e1u)}\n"
           f"{_poi_axis_label(p2)}: {_fmt(by, e2d, e2u)}\n"
           f"$\\rho = {corr:+.2f}$")
    ax.text(0.035, 0.965, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5, color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.45, ec="none"))

    ax.set_xlabel(_poi_axis_label(p1))
    ax.set_ylabel(_poi_axis_label(p2))
    cms_label_split(ax, cbar.ax, cfg)
    _stamp(ax, fit)
    save(fig, fitdir, f"nll_scan_{scan_tag(scan)}", console)


# ------------------------------------------------------------- driver


def render_fit_summary(entries: list, outdir: Path, var: str,
                       label: str | None = None, console=None) -> None:
    """Cross-fit POI summary from finished fit outputs. Top: each fit's
    per-category rate POIs vs the `var` window its category cut defines
    (e.g. ID SFs vs pT, one series per decay mode). Bottom: each fit's morph
    POIs (e.g. TES). Errors are the scan profile where a scan exists, else
    the FitDiagnostics Hesse error. `entries` = [(fit, cfg, fitdir), ...]."""
    from wham.expr import column_bounds, parse
    from wham.fitconfig import scale_entries
    from wham.render.common import set_style

    set_style()
    series = []  # (name, [(lo, hi, val, edn, eup)], [(morph, val, edn, eup)])
    for fit, cfg, fitdir in entries:
        path = fitdir / FITRESULT_FILE
        if not path.is_file():
            if console is not None:
                console.print(f"  [yellow]{fit.name}: no {FITRESULT_FILE} — run the fit first[/yellow]")
            continue
        params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
        windows = {c.name: column_bounds(parse(c.cut), var) if c.cut else (None, None)
                   for c in fit.categories}

        def interval(poi: str, fit=fit, fitdir=fitdir, params=params):
            """(value, err_dn, err_up): profile when scanned, else Hesse."""
            prof = _scan_poi_interval(fitdir, fit, poi)
            if prof is not None and np.isfinite(prof[1]) and np.isfinite(prof[2]):
                return prof
            p = params[poi]
            return p["value"], p["error"], p["error"]

        points = []
        for cat, _tmpl, expr in scale_entries(fit):
            poi = expr.strip()
            if cat is None or poi not in fit.model.pois or poi not in params:
                continue
            lo, hi = windows.get(cat, (None, None))
            if lo is None or hi is None:
                if console is not None:
                    console.print(f"  [yellow]{fit.name}/{cat}: no {var} window in "
                                  f"the category cut — skipped[/yellow]")
                continue
            points.append((lo, hi, *interval(poi)))
        morphs = [(m, *interval(m)) for m in fit.model.morphs if m in params]
        series.append((fit.name, sorted(points), morphs))

    if not series:
        return

    fig, (ax, axm) = plt.subplots(
        2, 1, figsize=(10, 10), height_ratios=[3, 2],
        gridspec_kw={"hspace": 0.3})
    n = len(series)
    for i, (name, points, _) in enumerate(series):
        if not points:
            continue
        lo = np.array([p[0] for p in points])
        hi = np.array([p[1] for p in points])
        centers = 0.5 * (lo + hi) + (i - (n - 1) / 2) * 0.05 * (hi - lo)
        vals = np.array([p[2] for p in points])
        errs = np.array([[p[3] for p in points], [p[4] for p in points]])
        ax.errorbar(centers, vals, xerr=[centers - lo, hi - centers], yerr=errs,
                    fmt="o", markersize=5, capsize=2, label=name)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, zorder=0)
    first_rate = next((f.model.pois for f, _, _ in entries if f.model.pois), None)
    ax.set_ylabel(_poi_axis_label(next(iter(first_rate))) if first_rate else "POI value")
    ax.set_xlabel(var_label(entries[0][1], var))
    ax.legend(fontsize=11)
    if label:
        ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=13)

    xm, labels_m = 0, []
    for i, (name, _, morphs) in enumerate(series):
        color = f"C{i}"
        for mname, val, edn, eup in morphs:
            axm.errorbar([xm], [val], yerr=[[edn], [eup]], fmt="o", color=color,
                         markersize=6, capsize=3)
            labels_m.append(mname)
            xm += 1
    axm.axhline(1.0, linestyle="--", color="gray", linewidth=1, zorder=0)
    axm.set_xticks(range(len(labels_m)), labels_m, fontsize=11)
    axm.set_xlim(-0.6, max(len(labels_m) - 0.4, 0.6))
    first_morph = next((f.model.morphs for f, _, _ in entries if f.model.morphs), None)
    axm.set_ylabel(_poi_axis_label(next(iter(first_morph))) if first_morph else "morph POI")
    cms_label(ax, entries[0][1])
    save(fig, outdir, "fitsummary", console)


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
                        xlabel=var_label(cfg, cat.variable), var=cat.variable,
                    )

    # ---- nuisance pulls + POI impacts, and the full POI summary
    _render_pulls(fit, cfg, fitdir, console)
    _render_poi_summary(fit, cfg, fitdir, console)

    # ---- NLL scans
    for scan in fit.resolved_scans():
        if len(scan.pois) == 1:
            _render_scan_1d(fit, cfg, fitdir, scan, console)
        else:
            _render_scan_2d(fit, cfg, fitdir, scan, console)
