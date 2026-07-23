"""Config-driven export of finished fit shapes to external-fitter inputs.

`wham export <yaml>` reads the shapes.root of one or more finished `wham fit`
runs and writes new ROOT files with renamed TDirectories/histograms, summed
processes, renamed systematics and optional rebinning — e.g. TauFW-Fitter
input files (see Configurations/tau_sf/*/taufw_export.yaml). The engine only
hardwires the wham shape conventions (nominal = plain process name,
systematics = `<proc>_<syst>Up/Down`, morph grids = `<proc>_<suffix><float>`);
every name, sum, rebin and output path lives in the export YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from wham import util
from wham.config import _Model

# ------------------------------------------------------------- config


class ExportBinCfg(_Model):
    from_: str = Field(alias="from")  # source shapes.root TDirectory (datacard bin)
    to: str                           # output TDirectory name
    rebin: int = Field(default=1, ge=1)  # merge factor (source nbins % rebin == 0)


class ExportSystCfg(_Model):
    """Systematic rename rule. `match` is the wham systematic name and may use
    `{bin}` (source bin name); `rename` is the output systematic name and may
    use `{bin}` and `{obin}` (output bin name). Up/Down suffixes are implied."""

    match: str
    rename: str


class ExportGridCfg(_Model):
    """Morph-grid passthrough: templates named `<proc>_<suffix><float>` follow
    their process under the output name, suffix string kept verbatim. The
    f == 1 point duplicates the nominal template and is skipped by default."""

    suffix: str = "TES"
    skip_nominal: bool = True


class ExportOutputCfg(_Model):
    file: Path                             # output ROOT file (abs or repo-relative)
    bins: list[ExportBinCfg] = Field(min_length=1)
    # output hist name -> input hist name(s); a list is SUMMED. A trailing '?'
    # marks an input optional (skipped if absent; output omitted if all are).
    processes: dict[str, str | list[str]] = Field(min_length=1)
    systematics: list[ExportSystCfg] = Field(default_factory=list)
    grids: list[ExportGridCfg] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "ExportOutputCfg":
        for field in ("from_", "to"):
            names = [getattr(b, field) for b in self.bins]
            dupes = {n for n in names if names.count(n) > 1}
            if dupes:
                raise ValueError(f"duplicate bin '{field.rstrip('_')}' names: {sorted(dupes)}")
        return self


class ExportConfig(_Model):
    name: str
    fits: list[str] = Field(min_length=1)  # fit YAMLs, relative to this file
    outputs: list[ExportOutputCfg] = Field(min_length=1)


def load_export_config(path: str | Path) -> ExportConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return ExportConfig.model_validate(raw)


# ------------------------------------------------------------- transformations


def parse_input(name: str) -> tuple[str, bool]:
    """'DY_2E?' -> ('DY_2E', True); plain name -> (name, False)."""
    if name.endswith("?"):
        return name[:-1], True
    return name, False


def rebin_hist(h: Any, factor: int) -> Any:
    """Merge `factor` adjacent bins (values AND variances summed)."""
    if factor == 1:
        return h
    import hist

    edges = h.axes[0].edges
    n = len(edges) - 1
    if n % factor:
        raise ValueError(f"cannot rebin {n} bins by a factor of {factor}")
    view = h.view()
    out = hist.Hist(hist.axis.Variable(edges[::factor]), storage=hist.storage.Weight())
    ov = out.view()
    ov["value"] = view["value"].reshape(-1, factor).sum(axis=1)
    ov["variance"] = view["variance"].reshape(-1, factor).sum(axis=1)
    return out


def sum_hists(hists: list[Any]) -> Any:
    out = hists[0].copy()
    for h in hists[1:]:
        out += h
    return out


def grid_float(name: str, proc: str, suffix: str) -> float | None:
    """'DY_genuine_TES0.950' with ('DY_genuine', 'TES') -> 0.95; None if
    `name` is not a grid template of `proc` (immediate-prefix match, so
    'tt_lfake_...' can never be mistaken for a grid point of 'tt')."""
    prefix = f"{proc}_{suffix}"
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):])
    except ValueError:
        return None


def transform_bin(src: dict[str, Any], out: ExportOutputCfg, b: ExportBinCfg,
                  notes: list[str]) -> dict[str, Any]:
    """All output hists for one output TDirectory, from one source dir.

    Sum rule: an output variant (systematic or grid point) exists if ANY input
    has it; inputs lacking it contribute their NOMINAL template. Variants are
    looked up by their expected names — source names are never globbed."""
    result: dict[str, Any] = {}
    for out_name, spec in out.processes.items():
        inputs = [spec] if isinstance(spec, str) else list(spec)
        present: list[str] = []
        for raw in inputs:
            name, optional = parse_input(raw)
            if name in src:
                present.append(name)
            elif optional:
                notes.append(f"{b.from_}: optional input '{name}' absent ('{out_name}')")
            else:
                raise KeyError(
                    f"bin '{b.from_}' has no histogram '{name}' (needed for '{out_name}')")
        if not present:
            notes.append(f"{b.from_}: '{out_name}' omitted (all inputs optional and absent)")
            continue

        result[out_name] = sum_hists([src[n] for n in present])

        for g in out.grids:
            # tail string -> {input: grid hist}; tails kept verbatim so the
            # output names reuse the source float formatting exactly
            points: dict[str, dict[str, Any]] = {}
            for name in present:
                head = len(name) + 1 + len(g.suffix)
                for key in src:
                    f = grid_float(key, name, g.suffix)
                    if f is None or (g.skip_nominal and abs(f - 1.0) < 1e-9):
                        continue
                    points.setdefault(key[head:], {})[name] = src[key]
            for tail, by_input in sorted(points.items()):
                result[f"{out_name}_{g.suffix}{tail}"] = sum_hists(
                    [by_input.get(n, src[n]) for n in present])

        for e in out.systematics:
            s = e.match.format(bin=b.from_)
            if not any(f"{n}_{s}Up" in src for n in present):
                continue
            oname = e.rename.format(bin=b.from_, obin=b.to)
            for ud in ("Up", "Down"):
                result[f"{out_name}_{oname}{ud}"] = sum_hists(
                    [src.get(f"{n}_{s}{ud}", src[n]) for n in present])

    if b.rebin > 1:
        result = {k: rebin_hist(h, b.rebin) for k, h in result.items()}
    return result


# ------------------------------------------------------------- I/O + driver


def read_shapes(path: Path) -> dict[str, dict[str, Any]]:
    """{TDirectory: {hist name: hist.Hist}} — Weight storage keeps Sumw2."""
    import uproot

    out: dict[str, dict[str, Any]] = {}
    with uproot.open(path) as f:
        for key, cls in f.classnames(recursive=True).items():
            if not cls.startswith("TH1"):
                continue
            k = key.split(";")[0]
            d, _, name = k.rpartition("/")
            if d:
                out.setdefault(d, {})[name] = f[key].to_hist()
    return out


def _compare_dirs(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    import numpy as np

    if set(a) != set(b):
        diff = sorted(set(a) ^ set(b))
        return f"histogram sets differ ({', '.join(diff[:4])}{'…' if len(diff) > 4 else ''})"
    for name in a:
        va, vb = a[name].view(), b[name].view()
        for field in ("value", "variance"):
            if not np.allclose(va[field], vb[field], rtol=1e-6, atol=1e-12):
                return f"'{name}' {field}s differ"
    return None


def pick_source(shapes_by_fit: dict[str, dict[str, dict[str, Any]]], bin_from: str,
                strict: bool = False, console: Any = None) -> dict[str, Any]:
    """The source dir for a bin. A bin present in several fits (e.g. a shared
    control bin) is consistency-checked; the first fit's copy is used."""
    hits = [(name, dirs[bin_from]) for name, dirs in shapes_by_fit.items()
            if bin_from in dirs]
    if not hits:
        raise KeyError(
            f"bin '{bin_from}' not found in any fit shapes "
            f"({', '.join(shapes_by_fit) or 'no fits'})")
    first_name, first = hits[0]
    for name, other in hits[1:]:
        msg = _compare_dirs(first, other)
        if msg is not None:
            full = f"bin '{bin_from}' differs between fits '{first_name}' and '{name}': {msg}"
            if strict:
                raise RuntimeError(full)
            if console is not None:
                console.print(f"[yellow]Warning:[/yellow] {full}")
    return first


def fit_shapes_path(ref: str, export_dir: Path) -> tuple[str, Path]:
    """(fit name, its shapes.root path) for a fit reference from the export
    YAML: a path, an export-dir-relative path, or a bare fit-config name."""
    from wham.combine import SHAPES_FILE
    from wham.fitconfig import load_fit_config, resolve_fit_config

    path = Path(ref)
    if not path.is_file():
        candidate = export_dir / ref
        path = candidate if candidate.is_file() else resolve_fit_config(ref)
    fit, cfg, _ = load_fit_config(path)
    return fit.name, cfg.resolved_output_dir() / "fit" / fit.name / SHAPES_FILE


def write_output(path: Path, dirs: dict[str, dict[str, Any]]) -> None:
    import uproot

    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        for bin_name, by_name in dirs.items():
            for name, h in by_name.items():
                clean = h.copy()
                flow = clean.view(flow=True)
                for field in ("value", "variance"):
                    flow[field][0] = 0.0
                    flow[field][-1] = 0.0
                f[f"{bin_name}/{name}"] = clean


def run_export(cfg: ExportConfig, config_dir: Path, *, strict: bool = False,
               console: Any = None) -> list[Path]:
    shapes_by_fit: dict[str, dict[str, dict[str, Any]]] = {}
    for ref in cfg.fits:
        name, spath = fit_shapes_path(ref, config_dir)
        if not spath.is_file():
            raise FileNotFoundError(
                f"no shapes for fit '{name}' ({spath}) — run `wham fit {ref}` first")
        shapes_by_fit[name] = read_shapes(spath)

    written: list[Path] = []
    for out in cfg.outputs:
        notes: list[str] = []
        dirs = {b.to: transform_bin(pick_source(shapes_by_fit, b.from_, strict, console),
                                    out, b, notes)
                for b in out.bins}
        target = out.file if out.file.is_absolute() else util.repo_root() / out.file
        write_output(target, dirs)
        written.append(target)
        if console is not None:
            nh = sum(len(v) for v in dirs.values())
            console.print(
                f"[bold green]Exported[/bold green] {len(dirs)} bins / {nh} hists -> {target}")
            for n in notes:
                console.print(f"  [dim]{n}[/dim]")
    return written
