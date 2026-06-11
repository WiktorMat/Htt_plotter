from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import wham.util

N_EVENTS = 4000


def _sample_table(rng: np.random.Generator, *, data: bool = False) -> pa.Table:
    n = N_EVENTS
    cols = {
        "pt_1": rng.uniform(10, 100, n),
        "pt_2": rng.uniform(10, 100, n),
        "eta_1": rng.uniform(-3, 3, n),
        "m_vis": rng.uniform(0, 250, n),
        "met_phi": rng.uniform(-np.pi, np.pi, n),
        "os": rng.integers(0, 2, n).astype(np.int32),
        "id_2": rng.integers(0, 8, n).astype(np.int32),
        "trg": rng.integers(0, 2, n).astype(np.int32),
        "weight": rng.uniform(0.5, 1.5, n),
        # tau_h-leg columns the BDT fake-factor models read (wham/muffin.py)
        "eta_2": rng.uniform(-2.5, 2.5, n),
        "phi_2": rng.uniform(-np.pi, np.pi, n),
        "charge_2": rng.choice([-1, 1], n).astype(np.int32),
        "decayMode_2": rng.choice([0, 1, 10, 11], n).astype(np.int32),
        "decayModePNet_2": rng.choice([0, 1, 10, 11], n).astype(np.int32),
        "jpt_2": rng.uniform(15, 120, n),
        "n_jets": rng.integers(0, 5, n).astype(np.int32),
        "n_bjets": rng.integers(0, 3, n).astype(np.int32),
        "met_pt": rng.uniform(0, 150, n),
        "met_dphi_2": rng.uniform(-np.pi, np.pi, n),
    }
    if not data:
        cols["wt_cp_sm"] = rng.uniform(0, 2, n)
        cols["wt_cp_ps"] = rng.uniform(0, 2, n)
    return pa.table(cols)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Synthetic data_dir with 2 MC + 1 data sample, isolated cache root."""
    monkeypatch.setattr(wham.util, "cache_root", lambda: tmp_path / "cache")

    rng = np.random.default_rng(11)
    data_dir = tmp_path / "data"
    names = {"TT_test": False, "DY_test": False, "Muon_test": True}
    for name, is_data in names.items():
        d = data_dir / name / "nominal"
        d.mkdir(parents=True)
        pq.write_table(_sample_table(rng, data=is_data), d / "merged.parquet")

    config_yaml = tmp_path / "analysis.yaml"
    config_yaml.write_text(
        f"""
name: test_analysis
lumi: 1000.0
data_dir: {data_dir}
output_dir: {tmp_path / "plots"}
selection: "pt_1 > 25 & abs(eta_1) < 2.4"
trigger: "trg == 1"
weight: weight
processes:
  QCD:  {{kind: qcd, color: "tab:olive"}}
  TT:   {{samples: ["TT_*"], color: "tab:purple"}}
  DY:   {{samples: ["DY_*"], color: "tab:orange"}}
  data: {{kind: data, samples: ["Muon_*"], color: black}}
qcd:
  method: abcd
  os: "os == 1"
  iso: "id_2 >= 5"
  antiiso: "id_2 > 1 & id_2 < 5"
sample_params:
  TT_test: {{xs: 100.0, eff: 50000}}
  DY_test: {{xs: 200.0, eff: 80000, filter_efficiency: 0.5}}
variables:
  pt_1:    {{bins: 20, range: [0, 100]}}
  pt_2:    {{bins: 20, range: [0, 100]}}
  m_vis:   {{bins: 25, range: [0, 250]}}
  met_phi: {{bins: 10, range: [-3.2, 3.2], kind: angle}}
plots:
  resolution: [[pt_1, pt_2], [m_vis, pt_1]]
  datamc: [pt_1, m_vis]
  cp:
    - {{var: met_phi, even: wt_cp_sm, odd: wt_cp_ps}}
""",
        encoding="utf-8",
    )
    return {"tmp": tmp_path, "yaml": config_yaml, "data_dir": data_dir}
