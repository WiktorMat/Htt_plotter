"""Tests for wham.export — config model, transformations, ROOT round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot

from wham.export import (
    ExportBinCfg,
    ExportConfig,
    ExportOutputCfg,
    grid_float,
    load_export_config,
    parse_input,
    pick_source,
    rebin_hist,
    run_export,
    sum_hists,
    transform_bin,
)


def _whist(values, variances=None, lo=40.0, hi=150.0):
    """Weight-storage 1D hist with explicit per-bin values/variances."""
    import hist

    values = np.asarray(values, dtype=float)
    variances = values.copy() if variances is None else np.asarray(variances, dtype=float)
    h = hist.Hist(
        hist.axis.Regular(len(values), lo, hi), storage=hist.storage.Weight()
    )
    view = h.view()
    view["value"] = values
    view["variance"] = variances
    return h


def _write_shapes(path: Path, dirs: dict[str, dict]) -> None:
    with uproot.recreate(path) as f:
        for d, by_name in dirs.items():
            for name, h in by_name.items():
                f[f"{d}/{name}"] = h


# ------------------------------------------------------------- pure helpers


def test_parse_input_optional() -> None:
    assert parse_input("DY_2E?") == ("DY_2E", True)
    assert parse_input("ST") == ("ST", False)


def test_rebin_pairwise() -> None:
    h = _whist(np.arange(22.0), variances=np.full(22, 0.25))
    out = rebin_hist(h, 2)
    assert len(out.axes[0].edges) - 1 == 11
    assert out.axes[0].edges[0] == 40.0 and out.axes[0].edges[-1] == 150.0
    assert np.allclose(out.view()["value"], np.arange(22.0).reshape(-1, 2).sum(axis=1))
    assert np.allclose(out.view()["variance"], 0.5)
    flow = out.view(flow=True)
    assert flow["value"][0] == 0.0 and flow["value"][-1] == 0.0


def test_rebin_identity_and_rejects_non_divisor() -> None:
    h = _whist(np.ones(22))
    assert rebin_hist(h, 1) is h
    with pytest.raises(ValueError, match="rebin 22 bins by a factor of 4"):
        rebin_hist(h, 4)


def test_sum_preserves_variances() -> None:
    a = _whist([1.0, 2.0], variances=[0.1, 0.2])
    b = _whist([3.0, 4.0], variances=[0.3, 0.4])
    out = sum_hists([a, b])
    assert np.allclose(out.view()["value"], [4.0, 6.0])
    assert np.allclose(out.view()["variance"], [0.4, 0.6])
    # inputs untouched
    assert np.allclose(a.view()["value"], [1.0, 2.0])


def test_grid_float() -> None:
    assert grid_float("DY_genuine_TES0.950", "DY_genuine", "TES") == pytest.approx(0.95)
    assert grid_float("DY_genuine_TES1.000", "DY_genuine", "TES") == pytest.approx(1.0)
    # not a grid point of this process
    assert grid_float("DY_genuine", "DY_genuine", "TES") is None
    assert grid_float("DY_genuine_ltf_dm0_pt1Up", "DY_genuine", "TES") is None
    # immediate-prefix rule: tt's grid never matches tt_lfake templates
    assert grid_float("tt_lfake_ltf_dm0_pt1Up", "tt", "TES") is None


# ------------------------------------------------------------- transform_bin


def _out_cfg(**over) -> ExportOutputCfg:
    base = {
        "file": "out.root",
        "bins": [{"from": "dm0_pt1", "to": "DM0_pt1"}],
        "processes": {"ZTT": "DY_genuine"},
    }
    base.update(over)
    return ExportOutputCfg.model_validate(base)


def test_transform_bin_renames_nominal() -> None:
    out = _out_cfg(processes={"ZTT": "DY_genuine", "data_obs": "data_obs"})
    src = {"DY_genuine": _whist([5.0, 7.0]), "data_obs": _whist([8.0, 9.0])}
    result = transform_bin(src, out, out.bins[0], notes=[])
    assert set(result) == {"ZTT", "data_obs"}
    assert np.allclose(result["ZTT"].view()["value"], [5.0, 7.0])


def test_transform_bin_sums_processes() -> None:
    out = _out_cfg(processes={"ST": ["ST", "ST_lfake", "ST_jfake"]})
    src = {"ST": _whist([1.0]), "ST_lfake": _whist([2.0]), "ST_jfake": _whist([4.0])}
    result = transform_bin(src, out, out.bins[0], notes=[])
    assert np.allclose(result["ST"].view()["value"], [7.0])


def test_transform_bin_required_missing_raises() -> None:
    out = _out_cfg(processes={"TTT": "tt"})
    with pytest.raises(KeyError, match=r"bin 'dm0_pt1' has no histogram 'tt'"):
        transform_bin({"DY_genuine": _whist([1.0])}, out, out.bins[0], notes=[])


def test_transform_bin_optional_missing_skipped() -> None:
    out = _out_cfg(processes={"ZJ": "DY_2E?", "ZL": "DY_2Mu"})
    notes: list[str] = []
    result = transform_bin({"DY_2Mu": _whist([3.0])}, out, out.bins[0], notes)
    assert set(result) == {"ZL"}
    assert any("DY_2E" in n for n in notes)


def test_transform_bin_syst_rename_and_nominal_fallback() -> None:
    out = _out_cfg(
        processes={"W": ["W_jets", "W_jfake"], "ZL": "DY_lfake"},
        systematics=[
            {"match": "jtf_{bin}", "rename": "shape_jTauFake_{obin}"},
            {"match": "ltf_{bin}", "rename": "shape_mTauFake_{obin}"},
        ],
    )
    src = {
        "W_jets": _whist([10.0]),
        "W_jfake": _whist([4.0]),
        "W_jfake_jtf_dm0_pt1Up": _whist([5.0]),
        "W_jfake_jtf_dm0_pt1Down": _whist([3.0]),
        "DY_lfake": _whist([2.0]),
        "DY_lfake_ltf_dm0_pt1Up": _whist([2.5]),
        "DY_lfake_ltf_dm0_pt1Down": _whist([1.5]),
    }
    result = transform_bin(src, out, out.bins[0], notes=[])
    # summed variant = shifted input + nominal of the untouched input
    assert np.allclose(result["W_shape_jTauFake_DM0_pt1Up"].view()["value"], [15.0])
    assert np.allclose(result["W_shape_jTauFake_DM0_pt1Down"].view()["value"], [13.0])
    assert np.allclose(result["ZL_shape_mTauFake_DM0_pt1Up"].view()["value"], [2.5])
    # no phantom variants: W has no ltf, ZL has no jtf
    assert "W_shape_mTauFake_DM0_pt1Up" not in result
    assert "ZL_shape_jTauFake_DM0_pt1Up" not in result


def test_transform_bin_grid_passthrough_skips_nominal() -> None:
    out = _out_cfg(grids=[{"suffix": "TES", "skip_nominal": True}])
    src = {"DY_genuine": _whist([9.0])}
    for f in ("0.950", "1.000", "1.050"):
        src[f"DY_genuine_TES{f}"] = _whist([float(f)])
    result = transform_bin(src, out, out.bins[0], notes=[])
    assert set(result) == {"ZTT", "ZTT_TES0.950", "ZTT_TES1.050"}
    assert np.allclose(result["ZTT_TES0.950"].view()["value"], [0.95])


def test_transform_bin_rebin_applies_to_variants() -> None:
    out = ExportOutputCfg.model_validate({
        "file": "out.root",
        "bins": [{"from": "dm0_pt5", "to": "DM0_pt5", "rebin": 2}],
        "processes": {"ZTT": "DY_genuine", "ZL": "DY_lfake"},
        "systematics": [{"match": "ltf_{bin}", "rename": "shape_mTauFake_{obin}"}],
        "grids": [{"suffix": "TES"}],
    })
    src = {
        "DY_genuine": _whist(np.ones(22)),
        "DY_genuine_TES0.950": _whist(np.ones(22)),
        "DY_lfake": _whist(np.ones(22)),
        "DY_lfake_ltf_dm0_pt5Up": _whist(np.ones(22)),
        "DY_lfake_ltf_dm0_pt5Down": _whist(np.ones(22)),
    }
    result = transform_bin(src, out, out.bins[0], notes=[])
    for name, h in result.items():
        assert len(h.axes[0].edges) - 1 == 11, name
        assert np.allclose(h.view()["value"], 2.0)


# ------------------------------------------------------------- config model


def test_config_validation() -> None:
    good = {
        "name": "x",
        "fits": ["a.yaml"],
        "outputs": [{"file": "o.root",
                     "bins": [{"from": "a", "to": "A"}],
                     "processes": {"P": "p"}}],
    }
    ExportConfig.model_validate(good)

    dup = {**good, "outputs": [{"file": "o.root",
                                "bins": [{"from": "a", "to": "A"},
                                         {"from": "b", "to": "A"}],
                                "processes": {"P": "p"}}]}
    with pytest.raises(ValueError, match="duplicate bin 'to'"):
        ExportConfig.model_validate(dup)

    with pytest.raises(ValueError):
        ExportBinCfg.model_validate({"from": "a", "to": "A", "rebin": 0})
    with pytest.raises(ValueError):  # unknown key (extra=forbid)
        ExportConfig.model_validate({**good, "nope": 1})
    with pytest.raises(ValueError):  # empty fits
        ExportConfig.model_validate({**good, "fits": []})


def test_load_export_config_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_export_config(p)


# ------------------------------------------------------------- sources + driver


def test_pick_source_consistency() -> None:
    zmm_a = {"tt": _whist([5.0])}
    zmm_b = {"tt": _whist([5.0])}
    by_fit = {"dm0": {"zmm": zmm_a}, "dm1": {"zmm": zmm_b}}
    assert pick_source(by_fit, "zmm") is zmm_a

    by_fit["dm1"]["zmm"] = {"tt": _whist([6.0])}
    with pytest.raises(RuntimeError, match="differs between fits 'dm0' and 'dm1'"):
        pick_source(by_fit, "zmm", strict=True)
    # non-strict: warns (no console here) and still returns the first copy
    assert pick_source(by_fit, "zmm") is zmm_a

    with pytest.raises(KeyError, match="bin 'nope' not found"):
        pick_source(by_fit, "nope")


def _fit_yaml(tmp_path: Path, name: str, category: str) -> Path:
    """Minimal fit+analysis YAML pair whose shapes.root path lands in tmp."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    ana = tmp_path / f"ana_{name}.yaml"
    ana.write_text(
        f"""
name: ana_{name}
lumi: 1.0
data_dir: {data}
output_dir: {tmp_path}/plots/ana_{name}
selection: "pt_1 > 0"
weight: weight
processes:
  DY: {{samples: ["DY*"], color: "tab:orange"}}
  data: {{kind: data, samples: ["Muon*"], color: "black"}}
variables:
  m_vis: {{bins: 2, range: [40, 150]}}
""",
        encoding="utf-8",
    )
    fit = tmp_path / f"{name}.yaml"
    fit.write_text(
        f"""
name: {name}
analysis: {ana}
categories:
  - {{name: {category}, variable: m_vis}}
model:
  pois:
    r: {{init: 1, range: [0, 3]}}
  processes:
    DY: {{scale: "r"}}
""",
        encoding="utf-8",
    )
    return fit


def test_run_export_roundtrip(tmp_path: Path) -> None:
    fit_a = _fit_yaml(tmp_path, "fita", "dm0_pt1")
    fit_b = _fit_yaml(tmp_path, "fitb", "dm1_pt1")
    shared = {"tt": _whist([5.0, 6.0]), "data_obs": _whist([7.0, 8.0])}
    for name, cat in (("fita", "dm0_pt1"), ("fitb", "dm1_pt1")):
        fitdir = tmp_path / "plots" / f"ana_{name}" / "fit" / name
        fitdir.mkdir(parents=True)
        _write_shapes(fitdir / "shapes.root", {
            cat: {"DY_genuine": _whist([1.0, 2.0], variances=[0.5, 0.5]),
                  "data_obs": _whist([3.0, 4.0])},
            "zmm": shared,
        })

    cfg = ExportConfig.model_validate({
        "name": "t",
        "fits": [str(fit_a), str(fit_b)],
        "outputs": [
            {"file": str(tmp_path / "new" / "mt.root"),
             "bins": [{"from": "dm0_pt1", "to": "DM0_pt1"},
                      {"from": "dm1_pt1", "to": "DM1_pt1"}],
             "processes": {"ZTT": "DY_genuine", "data_obs": "data_obs"}},
            {"file": str(tmp_path / "new" / "mm.root"),
             "bins": [{"from": "zmm", "to": "baseline"}],
             "processes": {"TT": "tt", "data_obs": "data_obs"}},
        ],
    })
    written = run_export(cfg, tmp_path, strict=True)
    assert [p.name for p in written] == ["mt.root", "mm.root"]

    with uproot.open(tmp_path / "new" / "mt.root") as f:
        keys = {k.split(";")[0] for k in f.keys()}
        assert {"DM0_pt1/ZTT", "DM0_pt1/data_obs", "DM1_pt1/ZTT"} <= keys
        h = f["DM0_pt1/ZTT"].to_hist()
        assert np.allclose(h.view()["value"], [1.0, 2.0])
        assert np.allclose(h.view()["variance"], [0.5, 0.5])  # Sumw2 round-trip
    with uproot.open(tmp_path / "new" / "mm.root") as f:
        assert np.allclose(f["baseline/TT"].to_hist().view()["value"], [5.0, 6.0])


def test_run_export_missing_shapes_errors(tmp_path: Path) -> None:
    fit_a = _fit_yaml(tmp_path, "fitc", "dm0_pt1")
    cfg = ExportConfig.model_validate({
        "name": "t", "fits": [str(fit_a)],
        "outputs": [{"file": str(tmp_path / "o.root"),
                     "bins": [{"from": "dm0_pt1", "to": "A"}],
                     "processes": {"P": "DY_genuine"}}],
    })
    with pytest.raises(FileNotFoundError, match="no shapes for fit 'fitc'"):
        run_export(cfg, tmp_path)


def test_taufw_export_configs_load() -> None:
    """Repo guard: the two TauFW export YAMLs stay loadable and consistent."""
    repo = Path(__file__).resolve().parents[3]
    paths = sorted(repo.glob("Configurations/tau_sf/*/taufw_export.yaml"))
    if len(paths) != 2:
        pytest.skip("taufw export configs not present")
    for p in paths:
        cfg = load_export_config(p)
        assert len(cfg.fits) == 5
        mt, mm = cfg.outputs
        assert len(mt.bins) == 25
        assert all(b.rebin == 2 for b in mt.bins if b.from_.endswith("_pt5"))
        assert mt.processes["ST"] == ["ST", "ST_lfake", "ST_jfake"]
        assert mm.bins[0].from_ == "zmm" and mm.bins[0].to == "baseline"
        assert "input_wham" in str(mt.file)  # never the reference input tree
