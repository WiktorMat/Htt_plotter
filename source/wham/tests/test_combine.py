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
    inject_toy_asymmetry,
    model_source,
    run_fit,
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


def test_dc_name_sanitization() -> None:
    assert dc_name("W+jets") == "W_jets"
    assert dc_name("DY_2Tau") == "DY_2Tau"


def test_resolve_fit_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_fit_config(str(tmp_path / "nope.yaml"))


# ------------------------------------------------------------- model source


def test_model_source_scale(fit_setup) -> None:
    src = model_source(fit_setup("scale")["fit"])
    assert 'self.modelBuilder.doVar("r[1,0,3]")' in src
    assert "SCALES = {'DY': 'r'}" in src  # bare POI: no expr:: needed
    assert 'self.modelBuilder.doSet("POI", "r")' in src


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
    s = fit_setup("scale")
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

    scan = uproot.open(fitdir / scan_file(fit, fit.resolved_scans()[0]))
    assert len(scan["limit"]["r"].array()) >= 10


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
