from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import uproot

from wham.combine import (
    dc_name,
    export_key,
    fit_templates,
    harvest_source,
    inject_toy_asymmetry,
    model_source,
    run_fit,
    scan_command,
    scan_file,
    write_datacard,
    write_shapes,
)
from wham.config import discover_samples
from wham.fill import fill_all
from wham.fitconfig import (
    FitConfig,
    SystematicCfg,
    category_analysis,
    fit_families,
    load_fit_config,
    resolve_fit_config,
    template_parents,
    translate_formula,
    union_analysis,
)
from wham.skim import ensure_skims

COMBINE_IMAGE = Path(
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/combine-standalone:latest"
)


def _cmssw_dir() -> str | None:
    """A CMSSW release with combine + CombineHarvester, for the morph backend."""
    import os

    for cand in (os.environ.get("CMSSW_BASE"),
                 "/afs/cern.ch/user/h/haawedik/CMSSW_14_1_0_pre4"):
        if cand and (Path(cand) / "src").is_dir():
            return cand
    return None


CMSSW_DIR = _cmssw_dir()

_SCALE_MODEL = """
model:
  pois:
    r: {init: 1, range: [0, 3]}
  processes:
    DY: {scale: "r"}
"""

_COMPONENTS_MODEL = """
model:
  pois:
    mu:    {init: 1, range: [0, 5]}
    alpha: {init: 0.7854, range: [0, 1.5708]}
  processes:
    DY:
      components:
        even: {weight: wt_cp_sm, scale: "mu * cos(alpha)^2"}
        odd:  {weight: wt_cp_ps, scale: "mu * sin(alpha)^2"}
"""

_ONE_CATEGORY = """
categories:
  - {name: SR, variable: met_phi}
"""

_TWO_CATEGORIES = """
categories:
  - {name: lowj,  variable: met_phi, cut: "n_jets < 2"}
  - {name: highj, variable: met_phi, cut: "n_jets >= 2"}
"""


@pytest.fixture
def fit_setup(workspace: dict):
    """New-schema fit YAML against the synthetic workspace + filled hists."""

    def _make(model: str = "scale", *, toy: float = 0.0, two_cats: bool = False,
              extra: str = "") -> dict:
        model_block = _SCALE_MODEL if model == "scale" else _COMPONENTS_MODEL
        cats_block = _TWO_CATEGORIES if two_cats else _ONE_CATEGORY
        fit_yaml = workspace["tmp"] / f"fit_{model}_{toy}_{two_cats}.yaml"
        fit_yaml.write_text(
            f"""
name: testfit_{model}{"_2cat" if two_cats else ""}
analysis: {workspace["yaml"]}
{cats_block}
{model_block}
asimov: {{enabled: true}}
toy: {{asymmetry: {toy}}}
scans:
  - {{pois: [{"r" if model == "scale" else "alpha"}], points: 10}}
systematics:
  - {{name: lumi, effect: lnN, processes: [TT, DY], scaleFactor: 1.025}}
  - {{name: xsec_tt, effect: lnN, processes: [TT], scaleFactor: 1.05}}
  - {{name: norm_qcd, effect: lnN, processes: [QCD], scaleFactor: 1.3}}
{extra}""",
            encoding="utf-8",
        )
        fit, cfg = load_fit_config(fit_yaml)
        samples, _ = discover_samples(cfg)
        skims = ensure_skims(union_analysis(fit, cfg), samples, workers=1)
        hists = {}
        for cat in fit.categories:
            ccfg = category_analysis(fit, cfg, cat)
            hists[cat.name] = fill_all(
                ccfg, samples, skims, families=fit_families(fit),
                only_vars=(cat.variable,), workers=1, sidecars=False,
            )
        return {"fit": fit, "cfg": cfg, "hists": hists, "samples": samples,
                "skims": skims, "tmp": workspace["tmp"], "yaml": fit_yaml}

    return _make


# ------------------------------------------------------------- formulas


def test_translate_formula() -> None:
    f, deps = translate_formula("mu * cos(alpha)^2", ["mu", "alpha"])
    assert f == "(@0*pow(cos(@1),2))"
    assert deps == ["mu", "alpha"]

    f, deps = translate_formula("r", ["r"])
    assert (f, deps) == ("@0", ["r"])

    f, deps = translate_formula("2*sin(a)*cos(a)", ["a"])
    assert f == "((2*sin(@0))*cos(@0))"
    assert deps == ["a"]

    # ** and ^ are equivalent
    assert translate_formula("x**2", ["x"]) == translate_formula("x^2", ["x"])


@pytest.mark.parametrize("bad", [
    "nope * 2",          # unknown name
    "max(r, 1)",         # disallowed function
    "r > 1",             # comparison
    "r.real",            # attribute access
    "pow(r, exponent=2)",  # keyword args
    "__import__('os')",  # nice try
    "r +",               # syntax error
])
def test_translate_formula_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid formula"):
        translate_formula(bad, ["r"])


# ------------------------------------------------------------- schema


def _fit_dict(**over) -> dict:
    base = {
        "name": "t",
        "analysis": "a",
        "categories": [{"name": "SR", "variable": "x"}],
        "model": {
            "pois": {"r": {"range": [0, 3]}},
            "processes": {"DY": {"scale": "r"}},
        },
    }
    base.update(over)
    return base


@pytest.mark.parametrize("over, msg", [
    ({"model": {"pois": {"r": {"range": [0, 3]}},
                "processes": {"DY": {"scale": "r", "components": {
                    "a": {"weight": "w", "scale": "r"}}}}}},
     "exactly one of"),
    ({"model": {"pois": {}, "processes": {"DY": {"scale": "r"}}}},
     "at least one POI"),
    ({"model": {"pois": {"r": {"range": [0, 3]}, "unused": {"range": [0, 1]}},
                "processes": {"DY": {"scale": "r"}}}},
     "not used by any scale"),
    ({"model": {"pois": {"r": {"range": [3, 0]}},
                "processes": {"DY": {"scale": "r"}}}},
     "must be increasing"),
    ({"scans": [{"pois": ["nope"], "points": 10}]}, "undeclared POIs"),
    ({"scans": [{"pois": ["r", "r"], "points": 10}]}, "duplicate POI"),
    ({"scans": [{"pois": ["r", "r2"], "points": 10, "range": [0, 1]}]},
     "1D scans only"),
    ({"scans": [{"pois": ["r"], "points": 10, "ranges": [[0, 1], [0, 1]]}]},
     "2D scans only"),
    ({"categories": [{"name": "SR", "variable": "x"},
                     {"name": "SR", "variable": "y"}]}, "duplicate category"),
    ({"categories": [{"name": "no spaces", "variable": "x"}]}, "datacard-safe"),
    ({"asimov": {"enabled": True, "parameters": {"nope": 1.0}}},
     "undeclared POIs"),
    ({"toy": {"asymmetry": 0.3}}, "exactly two components"),
    ({"systematics": [{"name": "s", "effect": "lnN", "processes": ["DY"],
                       "scaleFactor": 1.1, "categories": ["nope"]}]},
     "unknown"),
])
def test_fit_config_rejections(over: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        FitConfig.model_validate(_fit_dict(**over))


def test_systematic_cfg_validation() -> None:
    with pytest.raises(ValueError, match="needs a scaleFactor"):
        SystematicCfg(name="x", effect="lnN", processes=["A"])
    with pytest.raises(ValueError, match="no scaleFactor"):
        SystematicCfg(name="x", effect="rateParam", processes=["A"], scaleFactor=1.1)
    with pytest.raises(ValueError, match="rateParam-only"):
        SystematicCfg(name="x", effect="lnN", processes=["A"], scaleFactor=1.1,
                      range=(0.1, 5.0))
    s = SystematicCfg(name="x", effect="rateParam", processes=["A"], range=(0.1, 5.0))
    assert s.init == 1.0

    with pytest.raises(ValueError, match="either weight_up/weight_down or scales"):
        SystematicCfg(name="x", effect="shape", processes=["A"], weight_up="w")
    with pytest.raises(ValueError, match="do not apply to shape"):
        SystematicCfg(name="x", effect="shape", processes=["A"],
                      weight_up="w", weight_down="w2", scaleFactor=1.1)
    with pytest.raises(ValueError, match="shape-only"):
        SystematicCfg(name="x", effect="lnN", processes=["A"], scaleFactor=1.1,
                      weight_up="w")


def test_dc_name_sanitization() -> None:
    assert dc_name("W+jets") == "W_jets"
    assert dc_name("DY_2Tau") == "DY_2Tau"


def test_resolve_fit_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_fit_config(str(tmp_path / "nope.yaml"))


# ------------------------------------------------------------- model source


def test_scan_auto_window(fit_setup) -> None:
    fit = fit_setup("scale")["fit"]
    scan = fit.scans[0]  # 1D over r, no explicit range

    # no fit result yet -> full POI range
    cmd = scan_command(fit, scan, None)
    assert "--setParameterRanges r=0,3" in cmd

    # with a fit result -> best fit +- 10 sigma, clipped to the POI range
    fitresult = {"params": {"r": {"value": 1.0, "error": 0.02}}}
    cmd = scan_command(fit, scan, fitresult)
    assert "--setParameterRanges r=0.8,1.2" in cmd

    wide = {"params": {"r": {"value": 0.1, "error": 0.5}}}
    assert "--setParameterRanges r=0,3" in scan_command(fit, scan, wide)

    # explicit range always wins
    fixed = scan.model_copy(update={"range": (0.5, 1.5)})
    assert "--setParameterRanges r=0.5,1.5" in scan_command(fit, fixed, fitresult)


def test_model_source_scale(fit_setup) -> None:
    src = model_source(fit_setup("scale")["fit"])
    assert 'self.modelBuilder.doVar("r[1,0,3]")' in src
    assert "SCALES = {'DY': 'r'}" in src  # bare POI: no expr:: needed
    assert 'self.modelBuilder.doSet("POI", "r")' in src


def test_per_category_scale() -> None:
    from wham.fitconfig import scale_entries

    fit = FitConfig.model_validate({
        "name": "t", "analysis": "a",
        "categories": [{"name": "c1", "variable": "x"},
                       {"name": "c2", "variable": "x"}],
        "model": {
            "pois": {"sf1": {"range": [0, 3]}, "sf2": {"range": [0, 3]}},
            "processes": {"DY": {"scale": {"c1": "sf1", "c2": "sf2"}}},
        },
    })
    # one entry per (category, template), carrying its own expression
    assert set(scale_entries(fit)) == {("c1", "DY", "sf1"), ("c2", "DY", "sf2")}

    src = model_source(fit)
    assert 'self.modelBuilder.doVar("sf1[1,0,3]")' in src
    assert 'self.modelBuilder.doVar("sf2[1,0,3]")' in src
    # SCALES keyed by (bin, process); getYieldScale falls back per process
    assert "('c1', 'DY'): 'sf1'" in src
    assert "('c2', 'DY'): 'sf2'" in src
    assert "SCALES.get((bin, process)) or SCALES.get(process, 1)" in src
    assert 'self.modelBuilder.doSet("POI", "sf1,sf2")' in src


def test_per_category_scale_unknown_category() -> None:
    with pytest.raises(ValueError, match="scale maps unknown categories"):
        FitConfig.model_validate({
            "name": "t", "analysis": "a",
            "categories": [{"name": "c1", "variable": "x"}],
            "model": {
                "pois": {"sf1": {"range": [0, 3]}, "sf2": {"range": [0, 3]}},
                "processes": {"DY": {"scale": {"c1": "sf1", "nope": "sf2"}}},
            },
        })


def _morph_fit(**morph_over) -> dict:
    morph = {"process": "DY", "grid": {"from": 0.95, "to": 1.05, "step": 0.005},
             "categories": ["dm0_pt1"], "range": [0.95, 1.05],
             "scales": {"x": "sqrt", "pt_2": "linear"}}
    morph.update(morph_over)
    return {
        "name": "t", "analysis": "a",
        "categories": [{"name": "dm0_pt1", "variable": "x"},
                       {"name": "dm1_pt1", "variable": "x"}],
        "model": {
            "pois": {"sf0": {"range": [0, 3]}, "sf1": {"range": [0, 3]}},
            "processes": {"DY": {"scale": {"dm0_pt1": "sf0", "dm1_pt1": "sf1"}}},
            "morphs": {
                "tes_dm0": morph,
                "tes_dm1": {"process": "DY", "grid": {"from": 0.99, "to": 1.01, "step": 0.005},
                            "categories": ["dm1_pt1"], "range": [0.95, 1.05],
                            "scales": {"x": "sqrt"}},
            },
        },
    }


def test_morph_schema_and_variations() -> None:
    import math

    from wham.fitconfig import all_pois, grid_points, morph_point_label, morph_variations

    fit = FitConfig.model_validate(_morph_fit())
    assert all_pois(fit) == ["sf0", "sf1", "tes_dm0", "tes_dm1"]

    pts = grid_points(fit.model.morphs["tes_dm0"].grid)  # 'from' alias accepted
    assert len(pts) == 21 and pts[0] == 0.95 and pts[-1] == 1.05

    # per-category: only that category's morph, nominal (f=1) skipped
    v0 = morph_variations(fit, "dm0_pt1")
    assert len(v0) == 20
    assert all(x.target == "columns" and x.processes == ["DY"] for x in v0)
    by_name = {x.name: x for x in v0}
    lab = morph_point_label("tes_dm0", 1.005)
    # per-column laws: sqrt(f) for the mass-like observable, f for pt
    assert by_name[lab].factors["x"] == pytest.approx(math.sqrt(1.005))
    assert by_name[lab].factors["pt_2"] == pytest.approx(1.005)
    assert len(morph_variations(fit, "dm1_pt1")) == 4  # 5-point grid - nominal

    # default scans cover rate POIs then morph POIs
    assert [s.pois for s in fit.resolved_scans()] == [["sf0"], ["sf1"], ["tes_dm0"], ["tes_dm1"]]


def test_morph_needs_scales() -> None:
    with pytest.raises(ValueError):
        FitConfig.model_validate(_morph_fit(scales={}))


def test_morph_unknown_category() -> None:
    with pytest.raises(ValueError, match="morph 'tes_dm0' restricted to unknown categories"):
        FitConfig.model_validate(_morph_fit(categories=["nope"]))


def _morph_workspace_fit(workspace: dict):
    """Load a TES-morph fit against the synthetic workspace + fill its hists.

    POI is named `mu` (not `r`): the CombineHarvester morph backend reserves `r`
    for the default signal strength. cmssw is set so the CH backend can resolve a
    release; only the cmsenv integration test actually runs combine."""
    fit_yaml = workspace["tmp"] / "morphfit.yaml"
    fit_yaml.write_text(
        f"""
name: morphfit
analysis: {workspace["yaml"]}
categories:
  - {{name: SR, variable: met_phi}}
model:
  pois:
    mu: {{init: 1, range: [0, 3]}}
  processes:
    DY: {{scale: "mu"}}
  morphs:
    tes:
      process: DY
      grid: {{from: 0.9, to: 1.1, step: 0.05}}
      categories: [SR]
      scales: {{met_phi: sqrt}}
      range: [0.9, 1.1]
asimov: {{enabled: true}}
combine: {{cmssw: {CMSSW_DIR or "/nonexistent/CMSSW"}}}
scans:
  - {{pois: [tes], points: 8}}
""",
        encoding="utf-8",
    )
    fit, cfg = load_fit_config(fit_yaml)
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(union_analysis(fit, cfg), samples, workers=1)
    hists = {}
    for cat in fit.categories:
        ccfg = category_analysis(fit, cfg, cat)
        hists[cat.name] = fill_all(ccfg, samples, skims, families=fit_families(fit),
                                   only_vars=(cat.variable,), workers=1, sidecars=False)
    return fit, cfg, hists


def test_morph_harvest_export(workspace: dict) -> None:
    """A morph fit generates a CombineHarvester driver (cmsenv backend), not the
    container datacard. Checks the generated harvest.py + the TES grid shapes."""
    fit, cfg, hists = _morph_workspace_fit(workspace)
    templates, _signals, _bkgs = fit_templates(fit, cfg, hists)
    present = {c: set(t) for c, t in templates.items()}
    src = harvest_source(fit, cfg, present, template_parents(fit, cfg))

    # CMSHistFunc morph (not RooMomentMorph), the signal, the TES grid + POI names
    assert "BuildCMSHistFuncFactory" in src
    assert 'SIGNAL = "DY"' in src and 'TES = "tes"' in src
    assert "$BIN/$PROCESS_TES$MASS" in src           # CH signal shape pattern
    for mass in ("0.900", "1.000", "1.100"):
        assert f"'{mass}'" in src, mass
    assert "tid" not in src and "'name': 'mu'" in src  # the bare-POI scale -> rateParam

    # the 5-point grid (0.9..1.1 step .05) incl. the nominal point is in shapes.root
    shapes = workspace["tmp"] / "shapes_check.root"
    write_shapes(shapes, templates)
    with uproot.open(shapes) as f:
        names = {k.split("/")[-1].split(";")[0] for k in f.keys()}
    for tag in ("DY_TES0.900", "DY_TES1.000", "DY_TES1.100"):
        assert tag in names, tag


def test_model_source_components(fit_setup) -> None:
    src = model_source(fit_setup("components")["fit"])
    assert 'self.modelBuilder.doVar("mu[1,0,5]")' in src
    assert 'self.modelBuilder.doVar("alpha[0.7854,0,1.5708]")' in src
    assert "expr::scale_DY_even(\"(@0*pow(cos(@1),2))\", mu, alpha)" in src
    assert "expr::scale_DY_odd(\"(@0*pow(sin(@1),2))\", mu, alpha)" in src
    assert "'DY_even': 'scale_DY_even'" in src
    assert 'self.modelBuilder.doSet("POI", "mu,alpha")' in src


# ------------------------------------------------------------- templates


def test_templates_scale(fit_setup) -> None:
    s = fit_setup("scale")
    templates, signals, backgrounds = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert signals == ["DY"]
    t = templates["SR"]
    assert "data_obs" in t and "DY" in t and "TT" in t
    assert "TT" in backgrounds and "DY" not in backgrounds


def test_templates_components(fit_setup) -> None:
    s = fit_setup("components")
    templates, signals, backgrounds = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert signals == ["DY_even", "DY_odd"]
    t = templates["SR"]
    for name in signals:
        assert t[name].view()["value"].sum() > 0
    assert "DY" not in t and "TT" in backgrounds


def test_templates_two_categories(fit_setup) -> None:
    s = fit_setup("scale", two_cats=True)
    templates, signals, _ = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert set(templates) == {"lowj", "highj"}
    # the categories partition the events: their data_obs sums to the
    # single-category yield (same baseline selection)
    full = fit_setup("scale")
    total = fit_templates(full["fit"], full["cfg"], full["hists"])[0]
    got = sum(templates[c]["data_obs"].view()["value"].sum() for c in templates)
    want = total["SR"]["data_obs"].view()["value"].sum()
    assert got == pytest.approx(want, rel=1e-9)


def test_toy_injection() -> None:
    import hist

    h = hist.Hist(hist.axis.Regular(8, 0, 6.2832, name="x"), storage=hist.storage.Weight())
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 6.2832, 10000)
    h.fill(x=x, weight=np.ones_like(x))
    even, odd = h.copy(), h.copy()

    inject_toy_asymmetry(even, odd, 0.3)
    assert not np.allclose(even.view()["value"], odd.view()["value"])
    assert even.view()["value"].sum() == pytest.approx(h.view()["value"].sum(), rel=1e-12)
    assert odd.view()["value"].sum() == pytest.approx(h.view()["value"].sum(), rel=1e-12)
    assert np.all(even.view()["variance"] >= 0) and np.all(odd.view()["variance"] >= 0)


# ------------------------------------------------------------- writers


def test_shapes_roundtrip(fit_setup, tmp_path: Path) -> None:
    s = fit_setup("components", two_cats=True)
    templates, _, _ = fit_templates(s["fit"], s["cfg"], s["hists"])
    path = tmp_path / "shapes.root"
    write_shapes(path, templates)

    with uproot.open(path) as f:
        for cat, by_name in templates.items():
            for name, h1 in by_name.items():
                rh = f[f"{cat}/{name}"]
                assert np.allclose(rh.values(), h1.view()["value"]), (cat, name)
                assert np.allclose(rh.variances(), h1.view()["variance"]), (cat, name)


def test_datacard_multibin(fit_setup, tmp_path: Path) -> None:
    s = fit_setup(
        "scale", two_cats=True,
        extra=("  - {name: norm_tt, effect: rateParam, processes: [TT], range: [0.1, 5]}\n"
               "  - {name: only_low, effect: lnN, processes: [DY], scaleFactor: 1.1,\n"
               "     categories: [lowj]}"),
    )
    fit = s["fit"]
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    card = tmp_path / "datacard.txt"
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=template_parents(fit, s["cfg"]))
    text = card.read_text()
    rows = {l.split()[0]: l.split() for l in text.splitlines() if l.strip()}

    assert "imax 2" in text
    assert "shapes * * shapes.root $CHANNEL/$PROCESS" in text
    assert "* autoMCStats 10" in text

    # one column per (bin, process present there); ids consistent across bins
    bin_cells, proc_rows = None, []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "bin":
            bin_cells = parts[1:]  # last "bin" row = the column row
        if parts and parts[0] == "process":
            proc_rows.append(parts[1:])
    names, ids = proc_rows
    by_col = list(zip(bin_cells, names, ids))
    id_of = {}
    for b, n, i in by_col:
        assert id_of.setdefault(n, i) == i, "process id differs between bins"
    assert id_of["DY"] == "0"
    assert int(id_of["TT"]) > 0

    # rateParam: one line per bin, same (shared) parameter name
    rp = [l for l in text.splitlines() if l.startswith("norm_tt rateParam")]
    assert rp == [
        "norm_tt rateParam lowj TT 1 [0.1,5]",
        "norm_tt rateParam highj TT 1 [0.1,5]",
    ]
    # category-restricted lnN: applies in lowj, '-' in highj
    cells = dict(zip(zip(bin_cells, names), rows["only_low"][2:]))
    assert cells[("lowj", "DY")] == "1.1"
    assert cells[("highj", "DY")] == "-"


def test_datacard_parent_matching(fit_setup, tmp_path: Path) -> None:
    # patterns written against config names must hit sanitized datacard
    # columns ('T+T' -> 'T_T') and component templates via their parent
    s = fit_setup("components")
    fit = s["fit"].model_copy(update={"systematics": [
        SystematicCfg(name="xsec_dy", effect="lnN", processes=["DY"], scaleFactor=1.02),
    ]})
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    card = tmp_path / "datacard.txt"
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=template_parents(fit, s["cfg"]))
    text = card.read_text()
    row = next(l.split() for l in text.splitlines() if l.startswith("xsec_dy"))
    procs = [l.split()[1:] for l in text.splitlines() if l.split()[:1] == ["process"]][0]
    by_proc = dict(zip(procs, row[2:]))
    assert by_proc["DY_even"] == "1.02" and by_proc["DY_odd"] == "1.02"
    assert by_proc["TT"] == "-"


_TT_SHAPE = ('  - {name: tt_shape, effect: shape, processes: [TT],\n'
             '     weight_up: "weight * 2", weight_down: "weight * 0.5"}')


def test_shape_systematic_templates_and_datacard(fit_setup, tmp_path: Path) -> None:
    from wham.fitconfig import shape_affected

    s = fit_setup("scale", extra=_TT_SHAPE)
    fit = s["fit"]
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    t = templates["SR"]

    assert "TT_tt_shapeUp" in t and "TT_tt_shapeDown" in t
    assert t["TT_tt_shapeUp"].view()["value"].sum() == pytest.approx(
        2 * t["TT"].view()["value"].sum(), rel=1e-9)
    assert "DY_tt_shapeUp" not in t  # unmatched process: no varied template
    if "QCD" in t:  # the varied subtraction propagates into the QCD estimate
        assert "QCD_tt_shapeUp" in t

    card = tmp_path / "datacard.txt"
    syst = next(x for x in fit.systematics if x.effect == "shape")
    write_datacard(card, fit=fit, templates=templates, signal_names=signals,
                   background_names=backgrounds,
                   parents=template_parents(fit, s["cfg"]),
                   shape_affected_map={syst.name: shape_affected(fit, s["cfg"], syst)})
    text = card.read_text()
    row = next(l.split() for l in text.splitlines() if l.startswith("tt_shape"))
    assert row[1] == "shape"
    procs = [l.split()[1:] for l in text.splitlines() if l.split()[:1] == ["process"]][0]
    by_proc = dict(zip(procs, row[2:]))
    assert by_proc["TT"] == "1" and by_proc["DY"] == "-"

    # shapes roundtrip includes the Up/Down histograms
    shapes_path = tmp_path / "shapes.root"
    write_shapes(shapes_path, templates)
    with uproot.open(shapes_path) as f:
        assert np.allclose(f["SR/TT_tt_shapeUp"].values(),
                           t["TT_tt_shapeUp"].view()["value"])


def test_shape_resolution_errors(workspace: dict) -> None:
    fit_yaml = workspace["tmp"] / "bad_shape.yaml"
    fit_yaml.write_text(
        f"""
name: bad_shape
analysis: {workspace["yaml"]}
{_ONE_CATEGORY}
{_SCALE_MODEL}
systematics:
  - {{name: ff_shape, effect: shape, processes: [QCD],
     weight_up: "pt_2 / 50", weight_down: "pt_2 / 200"}}
""",
        encoding="utf-8",
    )
    # the synthetic analysis uses qcd.method=abcd
    with pytest.raises(ValueError, match="requires qcd.method=ff"):
        load_fit_config(fit_yaml)


def test_export_key_sensitivity(fit_setup) -> None:
    s0 = fit_setup("components", toy=0.0)
    s3 = fit_setup("components", toy=0.3)
    keys = {("SR", "datamc", "met_phi"): "k1", ("SR", "fitcp", "met_phi"): "k2"}
    assert export_key(s0["fit"], keys) != export_key(s3["fit"], keys)
    assert export_key(s0["fit"], keys) != export_key(
        s0["fit"], {**keys, ("SR", "datamc", "met_phi"): "OTHER"})
    assert export_key(s0["fit"], keys) == export_key(s0["fit"], dict(keys))


def test_datacard_only_skip_logic(fit_setup) -> None:
    s = fit_setup("scale")
    keys = {("SR", "datamc", "met_phi"): "k"}
    fitdir = run_fit(s["fit"], s["cfg"], s["hists"], keys, datacard_only=True)
    card = fitdir / "datacard.txt"
    assert (fitdir / "whammodel.py").is_file()
    assert (fitdir / "fitconfig.yaml").is_file()
    mtime = card.stat().st_mtime_ns

    # unchanged inputs -> export skipped
    run_fit(s["fit"], s["cfg"], s["hists"], keys, datacard_only=True)
    assert card.stat().st_mtime_ns == mtime

    # changed hist key -> re-export
    run_fit(s["fit"], s["cfg"], s["hists"],
            {("SR", "datamc", "met_phi"): "DIFFERENT"}, datacard_only=True)
    assert card.stat().st_mtime_ns != mtime


# ------------------------------------------------------------- container


@pytest.mark.combine
@pytest.mark.skipif(
    shutil.which("apptainer") is None or not COMBINE_IMAGE.exists(),
    reason="apptainer or combine image unavailable",
)
def test_full_rate_fit_in_container(fit_setup) -> None:
    s = fit_setup("scale", extra=_TT_SHAPE)
    fit = s["fit"]
    fitdir = run_fit(fit, s["cfg"], s["hists"], {("SR", "datamc", "met_phi"): "k"})

    assert (fitdir / "workspace.root").is_file()
    fd = uproot.open(fitdir / f"fitDiagnostics.{fit.name}.root")
    r_fit = fd["tree_fit_sb"]["r"].array()[0]
    assert r_fit == pytest.approx(1.0, abs=0.05)  # Asimov with r=1 injected

    result = json.loads((fitdir / "fitresult.json").read_text())
    assert result["params"]["r"]["value"] == pytest.approx(1.0, abs=0.05)
    assert result["params"]["r"]["error"] > 0
    assert "r" in result["impacts"]
    # the shape nuisance is fitted: Asimov pull ~ 0, constrained ~ 1
    assert result["params"]["tt_shape"]["value"] == pytest.approx(0.0, abs=0.2)
    assert 0.05 < result["params"]["tt_shape"]["error"] <= 1.2

    scan = uproot.open(fitdir / scan_file(fit, fit.resolved_scans()[0]))
    assert len(scan["limit"]["r"].array()) >= 10


@pytest.mark.combine
@pytest.mark.skipif(CMSSW_DIR is None, reason="no CMSSW with CombineHarvester available")
def test_morph_fit_cmsenv(workspace: dict) -> None:
    """Full TES-morph path through the CombineHarvester/cmsenv backend: the
    harvester builds the datacard with a CMSHistFunc morph (autoMCStats on),
    text2workspace + FitDiagnostics run, and the morph POI is floated/reported
    (physics recovery is covered separately; synthetic samples are featureless)."""
    fit, cfg, hists = _morph_workspace_fit(workspace)
    fitdir = run_fit(fit, cfg, hists, {("SR", "datamc", "met_phi"): "k"})

    # the CH backend writes its own datacard + shapes, then the workspace
    assert (fitdir / "harvest.py").is_file()
    assert (fitdir / "ch_shapes.root").is_file()
    assert (fitdir / "workspace.root").is_file()
    card = (fitdir / "datacard.txt").read_text()
    assert "autoMCStats" in card                           # BBB on (one-to-one postfit)

    result = json.loads((fitdir / "fitresult.json").read_text())
    assert "tes" in result["params"]                       # morph POI promoted + fitted
    assert result["params"]["tes"]["error"] >= 0
    assert result["params"]["mu"]["value"] == pytest.approx(1.0, abs=0.1)  # rate constrained
    assert (fitdir / scan_file(fit, fit.resolved_scans()[0])).is_file()   # tes scan ran


@pytest.mark.combine
@pytest.mark.skipif(
    shutil.which("apptainer") is None or not COMBINE_IMAGE.exists(),
    reason="apptainer or combine image unavailable",
)
def test_multipoi_two_category_recovery_in_container(fit_setup, workspace: dict) -> None:
    """3 POIs (mu, alpha + a TT rate param via model), 2 categories, toy
    asymmetry: the Asimov fit must recover the injected truth."""
    s = fit_setup(
        "components", toy=0.3, two_cats=True,
        extra="""
  - {name: norm_tt_free, effect: rateParam, processes: [TT], range: [0.1, 5]}
""",
    )
    # inject alpha=0.4 truth and add a 2D scan
    fit = s["fit"].model_copy(update={
        "asimov": s["fit"].asimov.model_copy(update={"parameters": {"alpha": 0.4}}),
        "scans": [
            s["fit"].scans[0],
            s["fit"].scans[0].model_copy(update={"pois": ["alpha", "mu"], "points": 8,
                                                 "range": None}),
        ],
    })
    keys = {(c.name, f, "met_phi"): f"k_{c.name}_{f}"
            for c in fit.categories for f in ("datamc", "fitcp")}
    fitdir = run_fit(fit, s["cfg"], s["hists"], keys)

    result = json.loads((fitdir / "fitresult.json").read_text())
    assert result["params"]["alpha"]["value"] == pytest.approx(0.4, abs=0.05)
    assert result["params"]["mu"]["value"] == pytest.approx(1.0, abs=0.05)
    assert result["params"]["norm_tt_free"]["value"] == pytest.approx(1.0, abs=0.1)
    assert result["impacts"]["alpha"]  # covariance impacts present

    # both scans produced output
    for scan in fit.resolved_scans():
        assert (fitdir / scan_file(fit, scan)).is_file(), scan.pois

    # datacard is genuinely multi-bin
    text = (fitdir / "datacard.txt").read_text()
    assert "imax 2" in text


def test_shape_systematic_scales_schema() -> None:
    from wham.fitconfig import SystematicCfg

    ok = SystematicCfg(name="jtf", effect="shape", processes=["DY_jfake"],
                       scales={"m_vis": "sqrt", "pt_2": "linear"}, shift=0.10)
    assert ok.shift == 0.10

    with pytest.raises(ValueError, match="either weight_up/weight_down or scales"):
        SystematicCfg(name="bad", effect="shape", processes=["DY"])  # neither flavor
    with pytest.raises(ValueError, match="either weight_up/weight_down or scales"):
        SystematicCfg(name="bad", effect="shape", processes=["DY"],
                      weight_up="w*2", weight_down="w*0.5",
                      scales={"pt_2": "linear"}, shift=0.03)  # both flavors
    with pytest.raises(ValueError, match="shape-only"):
        SystematicCfg(name="bad", effect="lnN", processes=["DY"], scaleFactor=1.1,
                      scales={"pt_2": "linear"}, shift=0.03)


def test_shape_variations_column_shift(workspace: dict) -> None:
    """A scales+shift shape systematic resolves into an up/down pair of
    columns-target fill variations with the per-law factors."""
    import math

    from wham.fitconfig import shape_variations

    fit_yaml = workspace["tmp"] / "esfit.yaml"
    fit_yaml.write_text(
        f"""
name: esfit
analysis: {workspace["yaml"]}
categories:
  - {{name: SR, variable: m_vis}}
model:
  pois:
    mu: {{init: 1, range: [0, 3]}}
  processes:
    DY: {{scale: "mu"}}
systematics:
  - {{name: jtf, effect: shape, processes: [TT],
      scales: {{m_vis: sqrt, pt_2: linear}}, shift: 0.10}}
""",
        encoding="utf-8",
    )
    fit, cfg = load_fit_config(fit_yaml)
    vars_ = shape_variations(fit, cfg)
    assert [v.name for v in vars_] == ["jtf_up", "jtf_down"]
    assert all(v.target == "columns" and v.processes == ["TT"] for v in vars_)
    up, down = vars_
    assert up.factors["pt_2"] == pytest.approx(1.10)
    assert up.factors["m_vis"] == pytest.approx(math.sqrt(1.10))
    assert down.factors["pt_2"] == pytest.approx(0.90)
    assert down.factors["m_vis"] == pytest.approx(math.sqrt(0.90))
    # slice labels line up with the datacard's {name}Up/Down template suffixes
    assert up.slice_labels() == ["jtf_up"] and down.slice_labels() == ["jtf_down"]
