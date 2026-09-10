from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

hist = pytest.importorskip("hist")

from wham.config import Sample, SampleParams, load_config
from wham.fill import build_fill_spec, fill_sample, merge_hists
from wham.jetfakes import compute_ff_fractions, estimate_jet_fakes
from wham.skim import SkimInfo


def _write_models(root: Path) -> Path:
    models = root / "models"
    for process in ("QCD", "Wjets", "ttbarMC"):
        mdir = models / f"model_mt_{process}"
        mdir.mkdir(parents=True)
        (mdir / "model.json").write_text("{}", encoding="utf-8")
    return models


def _config(tmp_path: Path, components: str) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models = _write_models(tmp_path)
    path = tmp_path / "analysis.yaml"
    path.write_text(f"""
name: jetfake_test
lumi: 1.0
data_dir: {data_dir}
output_dir: {tmp_path / "plots"}
selection: "pt_1 > 0"
trigger: "trg == 1"
weight: weight
processes:
  jet_fakes: {{kind: ff, color: "tab:olive"}}
  QCD: {{kind: qcd, color: "tab:olive"}}
  W_jfake:
    samples: ["W"]
    color: "tab:red"
    ff_component: Wjets
  tt_jfake:
    samples: ["TT"]
    color: "plum"
    ff_component: ttbar
  data:
    kind: data
    samples: ["Data"]
    color: black
qcd:
  method: abcd
  os: "os == 1"
  iso: "id_2 >= 7"
  antiiso: "id_2 >= 1 & id_2 < 7"
sample_params:
  W: {{xs: 1.0, eff: 1.0}}
  TT: {{xs: 1.0, eff: 1.0}}
fake_factors:
  models: {models}
  channel: mt
  era_label: 0
  estimate:
    enabled: true
    output_process: jet_fakes
    components: {components}
    application_region:
      selection: "os == 1"
      pass: "id_2 >= 7"
      fail: "id_2 >= 1 & id_2 < 7"
    fake_cut: "genPartFlav_2 == 0 | genPartFlav_2 == 6"
variables:
  x: {{bins: [0, 1]}}
plots:
  datamc: [x]
""", encoding="utf-8")
    return path


def _table(n_fail: int, *, wq: float, ww: float, wt: float, pass_weight: float = 0.0) -> pa.Table:
    n_pass = 1 if pass_weight else 0
    return pa.table({
        "pt_1": np.ones(n_fail + n_pass),
        "trg": np.ones(n_fail + n_pass, dtype=np.int32),
        "weight": np.r_[np.ones(n_fail), [pass_weight] if n_pass else []],
        "os": np.ones(n_fail + n_pass, dtype=np.int32),
        "id_2": np.r_[np.full(n_fail, 2, dtype=np.int32), [7] if n_pass else []],
        "x": np.full(n_fail + n_pass, 0.5),
        "genPartFlav_2": np.zeros(n_fail + n_pass, dtype=np.int32),
        "BDT_FF_score_QCD_sublead": np.full(n_fail + n_pass, wq),
        "BDT_FF_score_Wjets_sublead": np.full(n_fail + n_pass, ww),
        "BDT_FF_score_ttbarMC_sublead": np.full(n_fail + n_pass, wt),
    })


def _skim(tmp_path: Path, name: str, table: pa.Table) -> SkimInfo:
    path = tmp_path / f"{name}.parquet"
    pq.write_table(table, path)
    cols = frozenset(table.column_names)
    return SkimInfo(name, path, cols, cols, {}, table.num_rows)


def test_mixed_ff_fractions_and_nominal_prediction(tmp_path: Path) -> None:
    cfg = load_config(_config(
        tmp_path,
        "{QCD: true, Wjets: true, ttbar: true}",
    ))
    skims = {
        "Data": _skim(tmp_path, "Data", _table(1000, wq=0.02, ww=0.05, wt=0.03)),
        "W": _skim(tmp_path, "W", _table(300, wq=0.02, ww=0.05, wt=0.03, pass_weight=10.0)),
        "TT": _skim(tmp_path, "TT", _table(100, wq=0.02, ww=0.05, wt=0.03, pass_weight=20.0)),
    }
    samples = [
        Sample("Data", "data", "data", "nominal", skims["Data"].path, 1, 1, None),
        Sample("W", "W_jfake", "mc", "nominal", skims["W"].path, 1, 1,
               SampleParams(xs=1.0, eff=1.0)),
        Sample("TT", "tt_jfake", "mc", "nominal", skims["TT"].path, 1, 1,
               SampleParams(xs=1.0, eff=1.0)),
    ]

    fractions = compute_ff_fractions(cfg, samples, skims)
    assert fractions.fractions == pytest.approx({"QCD": 0.6, "Wjets": 0.3, "ttbar": 0.1})
    assert fractions.diagnostics["qcd_residual"] == pytest.approx(600.0)
    assert fractions.diagnostics["sum_fractions"] == pytest.approx(1.0)

    spec = build_fill_spec(cfg, families=["datamc"])
    spec = replace(
        spec,
        ffestimate_fractions=tuple(fractions.fractions.items()),
    )
    hists = {}
    for sample in samples:
        merge_hists(hists, fill_sample(sample, skims[sample.name], spec))
    estimate_jet_fakes(cfg, hists, fractions)

    h = hists[("datamc", "x")]
    jet = h[{"process": "jet_fakes", "region": "OS_iso", "variation": "nominal"}].view()
    w_fake = h[{"process": "W_jfake", "region": "OS_iso", "variation": "nominal"}].view()
    t_fake = h[{"process": "tt_jfake", "region": "OS_iso", "variation": "nominal"}].view()

    assert float(np.sum(jet["value"])) == pytest.approx(30.0)
    assert float(np.sum(jet["variance"])) == pytest.approx(1000 * 0.03**2)
    assert float(np.sum(w_fake["value"])) == 0.0
    assert float(np.sum(t_fake["value"])) == 0.0


def test_qcd_only_keeps_disabled_fake_mc_in_stack(tmp_path: Path) -> None:
    cfg = load_config(_config(
        tmp_path,
        "{QCD: true, Wjets: false, ttbar: false}",
    ))
    skims = {
        "Data": _skim(tmp_path, "Data", _table(1000, wq=0.02, ww=0.05, wt=0.03)),
        "W": _skim(tmp_path, "W", _table(300, wq=0.02, ww=0.05, wt=0.03, pass_weight=10.0)),
        "TT": _skim(tmp_path, "TT", _table(100, wq=0.02, ww=0.05, wt=0.03, pass_weight=20.0)),
    }
    samples = [
        Sample("Data", "data", "data", "nominal", skims["Data"].path, 1, 1, None),
        Sample("W", "W_jfake", "mc", "nominal", skims["W"].path, 1, 1,
               SampleParams(xs=1.0, eff=1.0)),
        Sample("TT", "tt_jfake", "mc", "nominal", skims["TT"].path, 1, 1,
               SampleParams(xs=1.0, eff=1.0)),
    ]
    fractions = compute_ff_fractions(cfg, samples, skims)
    spec = build_fill_spec(cfg, families=["datamc"])
    spec = replace(
        spec,
        ffestimate_fractions=tuple(fractions.fractions.items()),
    )
    hists = {}
    for sample in samples:
        merge_hists(hists, fill_sample(sample, skims[sample.name], spec))
    estimate_jet_fakes(cfg, hists, fractions)

    h = hists[("datamc", "x")]
    jet = h[{"process": "jet_fakes", "region": "OS_iso", "variation": "nominal"}].view()
    w_fake = h[{"process": "W_jfake", "region": "OS_iso", "variation": "nominal"}].view()
    t_fake = h[{"process": "tt_jfake", "region": "OS_iso", "variation": "nominal"}].view()

    assert fractions.fractions == {"QCD": 1.0}
    assert float(np.sum(jet["value"])) == pytest.approx((1000 - 300 - 100) * 0.02)
    assert float(np.sum(w_fake["value"])) == pytest.approx(10.0)
    assert float(np.sum(t_fake["value"])) == pytest.approx(20.0)
