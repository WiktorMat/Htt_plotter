from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wham.config import Sample, SampleParams, load_config
from wham.fill import build_fill_spec, fill_sample, hist_keys, make_hist, merge_hists
from wham.render.ffclosure import closure_metrics, render_ffclosure
from wham.skim import SkimInfo


def _write_minimal_model(root: Path) -> None:
    mdir = root / "model_mt_QCD"
    mdir.mkdir(parents=True)
    (mdir / "model.json").write_text("{}", encoding="utf-8")


def _analysis(tmp_path: Path, *, closure: str = "") -> Path:
    models = tmp_path / "models"
    _write_minimal_model(models)
    data_dir = tmp_path / "data"
    for name in ("Data_test", "TT_test"):
        (data_dir / name / "nominal").mkdir(parents=True)
        pq.write_table(pa.table({"x": [0.5]}), data_dir / name / "nominal" / "merged.parquet")
    path = tmp_path / "analysis.yaml"
    path.write_text(
        f"""
name: ffclosure_test
lumi: 1.0
data_dir: {data_dir}
output_dir: {tmp_path / "plots"}
selection: "pt_1 > 0"
trigger: "trg == 1"
weight: weight
processes:
  QCD:  {{kind: qcd, color: "tab:olive"}}
  TT:   {{samples: ["TT_*"], color: "tab:purple"}}
  data: {{kind: data, samples: ["Data_*"], color: black}}
qcd:
  method: abcd
  os: "os == 1"
  iso: "id_2 >= 7"
  antiiso: "id_2 >= 1 & id_2 < 7"
sample_params:
  TT_test: {{xs: 1.0, eff: 1.0}}
fake_factors:
  models: {models}
  channel: mt
  processes: [QCD]
  era_label: 0
{closure}
variables:
  x: {{bins: [0, 1, 2]}}
plots:
  datamc: [x]
""",
        encoding="utf-8",
    )
    return path


def _skim(tmp_path: Path, name: str, table: pa.Table) -> SkimInfo:
    path = tmp_path / f"{name}.parquet"
    pq.write_table(table, path)
    cols = frozenset(table.column_names)
    return SkimInfo(name, path, cols, cols, {}, table.num_rows)


def _samples(tmp_path: Path) -> list[Sample]:
    return [
        Sample(
            "Data_test",
            "data",
            "data",
            "nominal",
            tmp_path / "Data_test.parquet",
            1,
            1,
            None,
        ),
        Sample(
            "TT_test",
            "TT",
            "mc",
            "nominal",
            tmp_path / "TT_test.parquet",
            1,
            1,
            SampleParams(xs=1.0, eff=1.0),
        ),
    ]


def _closure_block(*, fail: str = "id_2 >= 1 & id_2 < 7") -> str:
    return f"""
  closure:
    enabled: true
    process: QCD
    selection: "os == 0"
    pass: "id_2 >= 7"
    fail: "{fail}"
    variables: [x]
"""


def test_no_closure_section_keeps_hist_families_unchanged(tmp_path: Path) -> None:
    cfg = load_config(_analysis(tmp_path))
    spec = build_fill_spec(cfg, families=["datamc"])
    assert [family for family, _, _ in hist_keys(spec)] == ["datamc"]


def test_ffclosure_target_prediction_sumw2_and_nan_diagnostics(tmp_path: Path) -> None:
    cfg = load_config(_analysis(tmp_path, closure=_closure_block()))
    data = pa.table({
        "pt_1": [1, 1, 1, 1, 1],
        "trg": [1, 1, 1, 1, 1],
        "weight": [1, 1, 1, 1, 1],
        "os": [0, 0, 0, 0, 0],
        "id_2": [7, 7, 2, 2, 2],
        "x": [0.5, 0.5, 0.5, 1.5, 0.5],
        "BDT_FF_score_QCD_sublead": [1.0, 1.0, 2.0, 3.0, np.nan],
    })
    tt = pa.table({
        "pt_1": [1, 1, 1, 1],
        "trg": [1, 1, 1, 1],
        "weight": [0.5, 1.0, 0.5, 1.0],
        "os": [0, 0, 0, 0],
        "id_2": [7, 7, 2, 2],
        "x": [0.5, 1.5, 0.5, 1.5],
        "BDT_FF_score_QCD_sublead": [1.0, 1.0, 2.0, 4.0],
    })
    skims = {
        "Data_test": _skim(tmp_path, "Data_test", data),
        "TT_test": _skim(tmp_path, "TT_test", tt),
    }
    spec = build_fill_spec(cfg, families=["ffclosure"])
    hists = {}
    for sample in _samples(tmp_path):
        merge_hists(hists, fill_sample(sample, skims[sample.name], spec))

    h = hists[("ffclosure", "x")]
    data_pass = h[{"process": "data", "region": "pass", "variation": "nominal"}].view()
    tt_pass = h[{"process": "TT", "region": "pass", "variation": "nominal"}].view()
    data_fail = h[{"process": "data", "region": "fail", "variation": "nominal"}].view()
    tt_fail = h[{"process": "TT", "region": "fail", "variation": "nominal"}].view()
    qcd_pass = h[{"process": "QCD", "region": "pass", "variation": "nominal"}].view()
    nan_weight = h[{"process": "data", "region": "nan_weight", "variation": "nominal"}].view()

    target = data_pass["value"] - tt_pass["value"] - qcd_pass["value"]
    prediction = data_fail["value"] - tt_fail["value"]
    target_var = data_pass["variance"] + tt_pass["variance"] + qcd_pass["variance"]
    prediction_var = data_fail["variance"] + tt_fail["variance"]

    assert np.allclose(target, [1.5, -1.0])
    assert np.allclose(prediction, [1.0, -1.0])
    assert np.allclose(target_var, [2.25, 1.0])
    assert np.allclose(prediction_var, [5.0, 25.0])
    assert float(np.sum(nan_weight["value"])) == 1.0

    metrics = closure_metrics(target, target_var, prediction, prediction_var, h.axes[-1].edges)
    assert metrics["norm_delta"] == pytest.approx(-1.0)
    assert metrics["shape_chi2_ndf"] is None
    assert metrics["max_abs_z"] == pytest.approx(0.5 / np.sqrt(7.25))
    assert metrics["target_negative_bins"] == 1
    assert metrics["prediction_negative_bins"] == 1


def test_ffclosure_rejects_overlapping_pass_fail(tmp_path: Path) -> None:
    cfg = load_config(_analysis(tmp_path, closure=_closure_block(fail="id_2 >= 1 & id_2 < 8")))
    table = pa.table({
        "pt_1": [1],
        "trg": [1],
        "weight": [1.0],
        "os": [0],
        "id_2": [7],
        "x": [0.5],
        "BDT_FF_score_QCD_sublead": [1.0],
    })
    skims = {"Data_test": _skim(tmp_path, "Data_test", table)}
    spec = build_fill_spec(cfg, families=["ffclosure"])

    with pytest.raises(RuntimeError, match="pass/fail selections overlap"):
        fill_sample(_samples(tmp_path)[0], skims["Data_test"], spec)


def test_render_ffclosure_creates_plot(tmp_path: Path) -> None:
    cfg = load_config(_analysis(tmp_path, closure=_closure_block()))
    spec = build_fill_spec(cfg, families=["ffclosure"])
    hists = {("ffclosure", "x"): make_hist(spec, "ffclosure", "x", spec.ffclosure[0][1])}
    h = hists[("ffclosure", "x")]
    view = h.view()
    procs = list(h.axes["process"])
    regions = list(h.axes["region"])
    view[procs.index("data"), regions.index("pass"), 0, :]["value"] = [2.0, 1.0]
    view[procs.index("data"), regions.index("pass"), 0, :]["variance"] = [2.0, 1.0]
    view[procs.index("data"), regions.index("fail"), 0, :]["value"] = [1.8, 1.2]
    view[procs.index("data"), regions.index("fail"), 0, :]["variance"] = [2.5, 1.5]

    render_ffclosure(cfg, hists, tmp_path / "plots")

    assert (tmp_path / "plots" / "ff_closure" / "x.png").is_file()
    assert (tmp_path / "plots" / "ff_closure" / "x.pdf").is_file()
