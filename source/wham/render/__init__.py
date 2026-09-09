"""Render dispatch."""

from __future__ import annotations

from pathlib import Path


def render_families(
    cfg,
    hists: dict,
    skims: dict,
    *,
    families: list[str],
    only_vars=None,
    outdir: Path,
    console=None,
) -> None:
    from wham.render.common import set_style

    set_style()
    outdir = Path(outdir)

    if "datamc" in families:
        from wham.render.datamc import render_datamc

        render_datamc(cfg, hists, outdir, only_vars, console)
    if "resolution" in families:
        from wham.render.resolution import render_resolution

        render_resolution(cfg, hists, outdir, only_vars, console)
    if "ffcheck" in families:
        from wham.render.ffcheck import render_ffcheck

        render_ffcheck(cfg, hists, outdir, only_vars, console)
    if "ffclosure" in families:
        from wham.render.ffclosure import render_ffclosure

        render_ffclosure(cfg, hists, outdir, only_vars, console)
    if "cp" in families:
        from wham.render.cp import render_cp

        render_cp(cfg, hists, outdir, only_vars, console)
    if "display3d" in families:
        from wham.render.display3d import render_display3d

        # display3d needs sample paths, not histograms
        from wham.config import discover_samples

        samples, _ = discover_samples(cfg)
        render_display3d(cfg, samples, outdir / "display3d", console)
