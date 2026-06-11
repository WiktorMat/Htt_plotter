"""Single-pass histogram filling from skims.

One hist.Hist per (family, variable) with identical axes everywhere:
    StrCategory(process) x StrCategory(region) x StrCategory(variation) x Regular(var)
    storage = Weight()  (value + variance, i.e. sumw2)

so per-sample partial hists merge across workers with plain `+`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wham.config import AnalysisConfig, Sample, VariableCfg, sample_scale
from wham.expr import parse
from wham.skim import SkimInfo, read_skim

REGION_NOMINAL = "nominal"
REGIONS_ABCD = ("OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso")
REGIONS_SS = ("OS", "SS")
REGIONS_CP = ("even", "odd")

HistKey = tuple[str, str]  # (family, variable-or-pair-name)


def resolution_name(reco: str, ref: str) -> str:
    return f"{reco}_from_{ref}"


def regions_for(family: str, qcd_method: str) -> tuple[str, ...]:
    if family == "datamc":
        return REGIONS_ABCD if qcd_method == "abcd" else REGIONS_SS
    if family in ("cp", "fitcp"):
        return REGIONS_CP
    return (REGION_NOMINAL,)


@dataclass(frozen=True)
class FillSpec:
    """Everything a worker needs, as plain picklable values."""

    selection: str
    trigger: str | None
    weight: str
    qcd_method: str
    qcd_os: str
    qcd_iso: str | None
    qcd_antiiso: str | None
    lumi: float
    processes: tuple[str, ...]            # category axis content, fixed order
    # requested fills
    resolution: tuple[tuple[str, str, dict, dict], ...]  # (reco, ref, rescfg, refcfg)
    datamc: tuple[tuple[str, dict], ...]
    cp: tuple[tuple[str, str, str, dict], ...]       # (var, even_col, odd_col, varcfg)
    # like cp but with signal-region cuts (trigger & os & iso), for fitting
    fitcp: tuple[tuple[str, str, str, dict], ...] = ()
    # variables whose source column differs from their name (alias -> column)
    var_columns: tuple[tuple[str, str], ...] = ()


def _axis(var: str, vcfg: dict):
    import hist

    bins = vcfg["bins"]
    if isinstance(bins, int):
        return hist.axis.Regular(
            bins, float(vcfg["range"][0]), float(vcfg["range"][1]), name=var
        )
    return hist.axis.Variable([float(e) for e in bins], name=var)


def make_hist(spec: FillSpec, family: str, var: str, vcfg: dict):
    import hist

    return hist.Hist(
        hist.axis.StrCategory(list(spec.processes), name="process"),
        hist.axis.StrCategory(list(regions_for(family, spec.qcd_method)), name="region"),
        hist.axis.StrCategory(["nominal"], name="variation"),
        _axis(var, vcfg),
        storage=hist.storage.Weight(),
    )


def resolution_binning(cfg: AnalysisConfig, reco: str, ref: str) -> dict:
    """Binning for the derived resolution variable.

    Uses an explicit `variables` entry named '<reco>_from_<ref>' when present,
    otherwise a sensible default for the formula used.
    """
    name = resolution_name(reco, ref)
    if name in cfg.variables:
        return cfg.variables[name].model_dump()
    ref_cfg = cfg.variables[ref]
    if ref_cfg.kind == "angle":
        return {"bins": 40, "range": (-np.pi, np.pi), "kind": "angle", "label": None,
                "relative": False}
    if not ref_cfg.relative:
        lo, hi = ref_cfg.span()
        span = hi - lo
        return {"bins": 40, "range": (-span / 2, span / 2), "kind": "scalar",
                "label": None, "relative": False}
    return {"bins": 40, "range": (-2.0, 2.0), "kind": "scalar", "label": None,
            "relative": True}


def build_fill_spec(
    cfg: AnalysisConfig,
    *,
    families: list[str],
    only_vars: tuple[str, ...] | None = None,
) -> FillSpec:
    def want(var: str) -> bool:
        return only_vars is None or var in only_vars

    def vdump(var: str) -> dict:
        return cfg.variables[var].model_dump()

    resolution = tuple(
        (reco, ref, resolution_binning(cfg, reco, ref), vdump(ref))
        for reco, ref in cfg.plots.resolution
        if "resolution" in families and (only_vars is None or want(reco) or want(ref))
    )
    datamc = tuple(
        (v, vdump(v)) for v in cfg.plots.datamc if "datamc" in families and want(v)
    )
    cp = tuple(
        (c.var, c.even, c.odd, vdump(c.var))
        for c in cfg.plots.cp
        if "cp" in families and want(c.var)
    )
    fitcp = tuple(
        (c.var, c.even, c.odd, vdump(c.var))
        for c in cfg.plots.fitcp
        if "fitcp" in families and want(c.var)
    )

    return FillSpec(
        selection=cfg.selection,
        trigger=cfg.trigger,
        weight=cfg.weight,
        qcd_method=cfg.qcd.method,
        qcd_os=cfg.qcd.os,
        qcd_iso=cfg.qcd.iso,
        qcd_antiiso=cfg.qcd.antiiso,
        lumi=cfg.lumi,
        processes=tuple(cfg.processes.keys()),
        resolution=resolution,
        datamc=datamc,
        cp=cp,
        fitcp=fitcp,
        var_columns=tuple(
            (v, vcfg.column) for v, vcfg in cfg.variables.items() if vcfg.column
        ),
    )


def hist_keys(spec: FillSpec) -> list[tuple[str, str, dict]]:
    """(family, name, binning) for every histogram the spec defines."""
    out: list[tuple[str, str, dict]] = []
    out += [
        ("resolution", resolution_name(reco, ref), rescfg)
        for reco, ref, rescfg, _ in spec.resolution
    ]
    out += [("datamc", v, vcfg) for v, vcfg in spec.datamc]
    out += [("cp", v, vcfg) for v, _, _, vcfg in spec.cp]
    out += [("fitcp", v, vcfg) for v, _, _, vcfg in spec.fitcp]
    return out


def spec_extras(spec: FillSpec) -> dict[HistKey, dict]:
    """Per-hist extra cache-key payload (CP weight columns, column aliases)."""
    extras: dict[HistKey, dict] = {}
    for var, even_col, odd_col, _ in spec.cp:
        extras[("cp", var)] = {"even": even_col, "odd": odd_col}
    for var, even_col, odd_col, _ in spec.fitcp:
        extras[("fitcp", var)] = {"even": even_col, "odd": odd_col}
    colmap = dict(spec.var_columns)
    for reco, ref, _, _ in spec.resolution:
        rc, fc = colmap.get(reco, reco), colmap.get(ref, ref)
        if (rc, fc) != (reco, ref):
            key = ("resolution", resolution_name(reco, ref))
            extras[key] = {**extras.get(key, {}), "columns": [rc, fc]}
    return extras


# ---------------------------------------------------------------- worker


class _Columns:
    """Numpy column access with caching; columns converted at most once."""

    def __init__(self, table: Any):
        self._table = table
        self._cache: dict[str, np.ndarray] = {}
        self.names = set(table.column_names)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def get(self, name: str) -> np.ndarray:
        arr = self._cache.get(name)
        if arr is None:
            arr = self._table.column(name).to_numpy(zero_copy_only=False)
            self._cache[name] = arr
        return arr

    def eval(self, source: str) -> np.ndarray:
        parsed = parse(source)
        return parsed.evaluate({c: self.get(c) for c in parsed.columns})

    def finite(self, name: str) -> np.ndarray:
        key = f"__finite__{name}"
        arr = self._cache.get(key)
        if arr is None:
            arr = np.isfinite(self.get(name))
            self._cache[key] = arr
        return arr


def fill_sample(sample: Sample, skim: SkimInfo, spec: FillSpec) -> dict[HistKey, Any]:
    """Fill every requested histogram for one sample. Runs in a worker."""
    selection = parse(spec.selection)
    needed = set(skim.columns)  # read everything the skim has; it is already minimal

    table = read_skim(skim, columns=sorted(needed), filter_expr=selection.arrow())
    cols = _Columns(table)
    n = table.num_rows

    scale = sample_scale(sample, spec.lumi)
    if sample.kind == "data":
        weights = np.ones(n)
    elif parse(spec.weight).columns <= cols.names:
        weights = cols.eval(spec.weight).astype(float) * scale
    else:
        weights = np.full(n, scale)

    hists: dict[HistKey, Any] = {}
    colmap = dict(spec.var_columns)

    def column(var: str) -> str:
        return colmap.get(var, var)

    def fill(family: str, name: str, vcfg: dict, region: str,
             values: np.ndarray, mask: np.ndarray, w: np.ndarray) -> None:
        key = (family, name)
        h = hists.get(key)
        if h is None:
            h = hists[key] = make_hist(spec, family, name, vcfg)
        h.fill(
            process=sample.process,
            region=region,
            variation="nominal",
            **{name: values[mask]},
            weight=w[mask],
        )

    # ---- resolution: derived variable
    for reco, ref, rescfg, refcfg in spec.resolution:
        reco_col, ref_col = column(reco), column(ref)
        if reco_col not in cols or ref_col not in cols:
            continue
        rv, cv = cols.get(reco_col), cols.get(ref_col)
        is_angle = refcfg.get("kind") == "angle"
        relative = bool(rescfg.get("relative", True)) and not is_angle
        mask = cols.finite(reco_col) & cols.finite(ref_col)
        if relative:
            mask = mask & (cv != 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                res = np.where(mask, (rv - cv) / np.where(cv == 0, 1.0, cv), 0.0)
        else:
            res = rv - cv
            if is_angle:
                res = (res + np.pi) % (2 * np.pi) - np.pi
        fill("resolution", resolution_name(reco, ref), rescfg,
             REGION_NOMINAL, res, mask, weights)

    # ---- region inputs shared by datamc and fitcp (each evaluated at most once)
    _parts: dict[str, np.ndarray | None] = {}

    def region_part(name: str) -> np.ndarray | None:
        """Boolean mask for trigger/os/iso/anti, or None if undefined/missing cols."""
        if name not in _parts:
            src = {
                "trigger": spec.trigger,
                "os": spec.qcd_os,
                "iso": spec.qcd_iso,
                "anti": spec.qcd_antiiso,
            }[name]
            if src is not None and parse(src).columns <= cols.names:
                _parts[name] = cols.eval(src).astype(bool)
            else:
                _parts[name] = None
        return _parts[name]

    def base_mask() -> np.ndarray:
        trig = region_part("trigger")
        return trig if trig is not None else np.ones(n, dtype=bool)

    def sr_mask() -> np.ndarray | None:
        """Signal region: trigger & OS (& iso for abcd)."""
        os_mask = region_part("os")
        if os_mask is None:
            return None
        sr = base_mask() & os_mask
        if spec.qcd_method == "abcd":
            iso = region_part("iso")
            if iso is None:
                return None
            sr = sr & iso
        return sr

    # ---- datamc: trigger x charge x isolation regions
    if spec.datamc and region_part("os") is not None:
        base = base_mask()
        os_mask = region_part("os")

        region_masks: dict[str, np.ndarray] = {}
        if spec.qcd_method == "abcd":
            iso = region_part("iso")
            anti = region_part("anti")
            if iso is not None and anti is not None:
                region_masks = {
                    "OS_iso": base & os_mask & iso,
                    "SS_iso": base & ~os_mask & iso,
                    "OS_antiiso": base & os_mask & anti,
                    "SS_antiiso": base & ~os_mask & anti,
                }
        else:
            region_masks = {"OS": base & os_mask, "SS": base & ~os_mask}

        for var, vcfg in spec.datamc:
            col = column(var)
            if col not in cols:
                continue
            values = cols.get(col)
            finite = cols.finite(col)
            for region, rmask in region_masks.items():
                fill("datamc", var, vcfg, region, values, rmask & finite, weights)

    # ---- cp: even/odd CP weights (MC only)
    if sample.kind != "data":
        for var, even_col, odd_col, vcfg in spec.cp:
            col = column(var)
            if col not in cols or even_col not in cols or odd_col not in cols:
                continue
            values = cols.get(col)
            finite = cols.finite(col)
            fill("cp", var, vcfg, "even", values, finite, weights * cols.get(even_col))
            fill("cp", var, vcfg, "odd", values, finite, weights * cols.get(odd_col))

    # ---- fitcp: CP hypothesis templates in the signal region (MC only)
    if spec.fitcp and sample.kind != "data":
        sr = sr_mask()
        if sr is not None:
            for var, even_col, odd_col, vcfg in spec.fitcp:
                col = column(var)
                if col not in cols or even_col not in cols or odd_col not in cols:
                    continue
                values = cols.get(col)
                mask = sr & cols.finite(col)
                fill("fitcp", var, vcfg, "even", values, mask, weights * cols.get(even_col))
                fill("fitcp", var, vcfg, "odd", values, mask, weights * cols.get(odd_col))

    return hists


def merge_hists(into: dict[HistKey, Any], part: dict[HistKey, Any]) -> None:
    for key, h in part.items():
        if key in into:
            into[key] = into[key] + h
        else:
            into[key] = h


# ---------------------------------------------------------------- driver


def fill_all(
    cfg: AnalysisConfig,
    samples: list[Sample],
    skims: dict[str, SkimInfo],
    *,
    families: list[str],
    only_vars: tuple[str, ...] | None = None,
    workers: int = 6,
    use_cache: bool = True,
    cache_only: bool = False,
    console=None,
) -> dict[HistKey, Any]:
    """Return all requested histograms, filling only what the cache lacks."""
    from wham.histcache import cache_key, load_hist, save_hist, write_sidecars
    from wham.parallel import run_parallel
    from wham.qcd import estimate_qcd

    spec = build_fill_spec(cfg, families=families, only_vars=only_vars)
    wanted = hist_keys(spec)
    extras = spec_extras(spec)

    def _key(family: str, name: str, vcfg: dict) -> str:
        return cache_key(cfg, samples, skims, spec, family, name, vcfg,
                         extra=extras.get((family, name)))

    hists: dict[HistKey, Any] = {}
    missing: list[tuple[str, str, dict]] = []
    for family, name, vcfg in wanted:
        key = _key(family, name, vcfg)
        cached = load_hist(cfg.name, family, name, key) if use_cache else None
        if cached is not None:
            hists[(family, name)] = cached
        else:
            missing.append((family, name, vcfg))

    if missing and cache_only:
        names = ", ".join(f"{f}/{n}" for f, n, _ in missing)
        raise RuntimeError(
            f"histogram cache is stale or missing for: {names}. Run `wham plot`."
        )

    if missing:
        # One pass over the skims fills every missing histogram at once.
        missing_keys = {(f, n) for f, n, _ in missing}
        sub_spec = _restrict_spec(spec, missing_keys)
        jobs = [(s, skims[s.name], sub_spec) for s in samples if s.name in skims]
        fresh: dict[HistKey, Any] = {}
        done = 0
        for part in run_parallel(fill_sample, jobs, workers=workers,
                                 size_of=lambda j: j[1].path.stat().st_size):
            merge_hists(fresh, part)
            done += 1
            if console is not None:
                console.print(f"  filled {done}/{len(jobs)} samples", end="\r")

        # Samples can skip hists whose columns they lack; materialize empties.
        for family, name, vcfg in missing:
            if (family, name) not in fresh:
                fresh[(family, name)] = make_hist(sub_spec, family, name, vcfg)

        if "datamc" in families:
            estimate_qcd(cfg, {k: h for k, h in fresh.items() if k[0] == "datamc"})

        for family, name, vcfg in missing:
            h = fresh[(family, name)]
            save_hist(cfg.name, family, name, _key(family, name, vcfg), h)
            hists[(family, name)] = h

        write_sidecars(cfg, hists)

    return hists


def _restrict_spec(spec: FillSpec, keys: set[HistKey]) -> FillSpec:
    from dataclasses import replace

    return replace(
        spec,
        resolution=tuple(
            x for x in spec.resolution
            if ("resolution", resolution_name(x[0], x[1])) in keys
        ),
        datamc=tuple(x for x in spec.datamc if ("datamc", x[0]) in keys),
        cp=tuple(x for x in spec.cp if ("cp", x[0]) in keys),
        fitcp=tuple(x for x in spec.fitcp if ("fitcp", x[0]) in keys),
    )
