from __future__ import annotations

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
    run_fit,
    write_datacard,
    write_shapes,
)
from wham.config import discover_samples
from wham.fill import fill_all
from wham.fitconfig import augmented_analysis, load_fit_config, resolve_fit_config
from wham.skim import ensure_skims

COMBINE_IMAGE = Path(
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/combine-standalone:latest"
)


@pytest.fixture
def fit_setup(workspace: dict):
    """Fit YAML against the synthetic workspace + filled histograms."""

    def _make(mode: str = "cp", toy: float = 0.0, extra: str = "") -> dict:
        fit_yaml = workspace["tmp"] / f"fit_{mode}_{toy}.yaml"
        fit_yaml.write_text(
            f"""
name: testfit_{mode}
analysis: {workspace["yaml"]}
variable: met_phi
mode: {mode}
signal: DY
bin: SR
asimov: {{enabled: true}}
toy: {{asymmetry: {toy}}}
scan: {{points: 10}}
systematics:
  - {{name: lumi, effect: lnN, processes: [TT, DY], scaleFactor: 1.025}}
  - {{name: xsec_tt, effect: lnN, processes: [TT], scaleFactor: 1.05}}
  - {{name: norm_qcd, effect: lnN, processes: [QCD], scaleFactor: 1.3}}
{extra}""",
            encoding="utf-8",
        )
        fit, cfg = load_fit_config(fit_yaml)
        aug, families = augmented_analysis(fit, cfg)
        samples, _ = discover_samples(aug)
        skims = ensure_skims(aug, samples, workers=1)
        hists = fill_all(aug, samples, skims, families=families,
                         only_vars=(fit.variable,), workers=1)
        return {"fit": fit, "cfg": aug, "hists": hists, "samples": samples,
                "skims": skims, "tmp": workspace["tmp"]}

    return _make


def test_templates_cp_mode(fit_setup) -> None:
    s = fit_setup("cp")
    templates, signals, backgrounds = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert signals == ["DY_cpeven", "DY_cpodd"]
    assert "data_obs" in templates
    assert "TT" in backgrounds and "DY" not in backgrounds
    # QCD may be dropped if ABCD yields zero on synthetic data; fine either way
    for name in signals:
        assert templates[name].view()["value"].sum() > 0


def test_templates_rate_mode(fit_setup) -> None:
    s = fit_setup("rate")
    templates, signals, backgrounds = fit_templates(s["fit"], s["cfg"], s["hists"])
    assert signals == ["DY"]
    assert "DY" in templates and "DY" not in backgrounds


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


def test_shapes_roundtrip(fit_setup, tmp_path: Path) -> None:
    s = fit_setup("cp")
    templates, _, _ = fit_templates(s["fit"], s["cfg"], s["hists"])
    path = tmp_path / "shapes.root"
    write_shapes(path, "SR", templates)

    with uproot.open(path) as f:
        for name, h1 in templates.items():
            rh = f[f"SR/{name}"]
            assert np.allclose(rh.values(), h1.view()["value"]), name
            assert np.allclose(rh.variances(), h1.view()["variance"]), name


def test_datacard_text(fit_setup, tmp_path: Path) -> None:
    s = fit_setup("cp")
    fit = s["fit"]
    templates, signals, backgrounds = fit_templates(fit, s["cfg"], s["hists"])
    card = tmp_path / "datacard.txt"
    write_datacard(card, fit=fit, bin_name="SR", templates=templates,
                   signal_names=signals, background_names=backgrounds)
    text = card.read_text()

    assert "shapes * SR shapes.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC" in text
    assert f"observation  {templates['data_obs'].view()['value'].sum():.4f}" in text
    assert "SR autoMCStats 10" in text

    lines = {l.split()[0]: l.split() for l in text.splitlines() if l.strip()}
    procs = None
    ids = None
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "process":
            if procs is None:
                procs = parts[1:]
            else:
                ids = [int(x) for x in parts[1:]]
    assert procs[:2] == ["DY_cpeven", "DY_cpodd"]
    assert ids[:2] == [0, -1]
    assert all(i > 0 for i in ids[2:])

    # lumi hits both split signals (parent DY) and TT; not QCD
    lumi = lines["lumi"]
    by_proc = dict(zip(procs, lumi[2:]))
    assert by_proc["DY_cpeven"] == "1.025" and by_proc["DY_cpodd"] == "1.025"
    assert by_proc["TT"] == "1.025"
    if "QCD" in by_proc:
        assert by_proc["QCD"] == "-"


def test_export_key_sensitivity(fit_setup) -> None:
    s0 = fit_setup("cp", toy=0.0)
    s3 = fit_setup("cp", toy=0.3)
    keys = {("datamc", "met_phi"): "k1", ("fitcp", "met_phi"): "k2"}
    assert export_key(s0["fit"], keys) != export_key(s3["fit"], keys)
    assert export_key(s0["fit"], keys) != export_key(s0["fit"], {**keys, ("datamc", "met_phi"): "OTHER"})
    assert export_key(s0["fit"], keys) == export_key(s0["fit"], dict(keys))


def test_datacard_only_skip_logic(fit_setup) -> None:
    s = fit_setup("cp")
    fitdir = run_fit(s["fit"], s["cfg"], s["hists"], {("a", "b"): "k"},
                     datacard_only=True)
    card = fitdir / "datacard.txt"
    mtime = card.stat().st_mtime_ns

    # unchanged inputs -> export skipped
    run_fit(s["fit"], s["cfg"], s["hists"], {("a", "b"): "k"}, datacard_only=True)
    assert card.stat().st_mtime_ns == mtime

    # changed hist key -> re-export
    run_fit(s["fit"], s["cfg"], s["hists"], {("a", "b"): "DIFFERENT"}, datacard_only=True)
    assert card.stat().st_mtime_ns != mtime


def test_dc_name_sanitization() -> None:
    assert dc_name("W+jets") == "W_jets"
    assert dc_name("DY_2Tau") == "DY_2Tau"


def test_resolve_fit_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_fit_config(str(tmp_path / "nope.yaml"))


@pytest.mark.combine
@pytest.mark.skipif(
    shutil.which("apptainer") is None or not COMBINE_IMAGE.exists(),
    reason="apptainer or combine image unavailable",
)
def test_full_rate_fit_in_container(fit_setup) -> None:
    s = fit_setup("rate")
    fit = s["fit"]
    keys = {("datamc", "met_phi"): "k"}
    fitdir = run_fit(fit, s["cfg"], s["hists"], keys)

    assert (fitdir / "workspace.root").is_file()
    fd = uproot.open(fitdir / f"fitDiagnostics.{fit.name}.root")
    r_fit = fd["tree_fit_sb"]["r"].array()[0]
    assert r_fit == pytest.approx(1.0, abs=0.05)  # Asimov with r=1 injected

    scan = uproot.open(fitdir / f"higgsCombine.scan_{fit.name}.MultiDimFit.mH120.root")
    assert len(scan["limit"]["r"].array()) >= fit.scan.points
