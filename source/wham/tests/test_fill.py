from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from wham.config import discover_samples, load_config, sample_scale
from wham.fill import (
    build_fill_spec,
    fill_all,
    fill_sample,
    merge_hists,
    resolution_name,
)
from wham.skim import ensure_skims

FAMILIES = ["resolution", "datamc", "cp"]


@pytest.fixture
def filled(workspace: dict) -> dict:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=FAMILIES)

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))
    return {"cfg": cfg, "samples": samples, "skims": skims, "spec": spec, "hists": hists}


def _reference_frame(sample, cfg):
    df = pq.read_table(sample.path).to_pandas()
    sel = (
        (df.pt_1 > 25) & (df.eta_1.abs() < 2.4)
    )
    df = df[sel].copy()
    scale = sample_scale(sample, cfg.lumi)
    df["w"] = 1.0 if sample.kind == "data" else df.weight * scale
    return df


def test_datamc_regions_partition(filled: dict) -> None:
    cfg = filled["cfg"]
    h = filled["hists"][("datamc", "m_vis")]
    sample = next(s for s in filled["samples"] if s.kind == "data")
    df = _reference_frame(sample, cfg)
    df = df[(df.trg == 1) & (df.m_vis >= 0) & (df.m_vis < 250)]

    expectations = {
        "OS_iso": (df.os == 1) & (df.id_2 >= 5),
        "SS_iso": (df.os == 0) & (df.id_2 >= 5),
        "OS_antiiso": (df.os == 1) & (df.id_2 > 1) & (df.id_2 < 5),
        "SS_antiiso": (df.os == 0) & (df.id_2 > 1) & (df.id_2 < 5),
    }
    for region, mask in expectations.items():
        got = float(
            h[{"process": "data", "region": region, "variation": "nominal"}]
            .view()["value"].sum()
        )
        assert got == pytest.approx(float(mask.sum())), region


def test_resolution_relative_formula(filled: dict) -> None:
    cfg = filled["cfg"]
    name = resolution_name("pt_1", "pt_2")
    h = filled["hists"][("resolution", name)]
    sample = next(s for s in filled["samples"] if s.name == "TT_test")
    df = _reference_frame(sample, cfg)
    res = (df.pt_1 - df.pt_2) / df.pt_2
    in_range = (res >= -2) & (res < 2) & (df.pt_2 != 0)
    expected = df.w[in_range].sum()
    got = float(
        h[{"process": "TT", "region": "nominal", "variation": "nominal"}].view()["value"].sum()
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_cp_even_odd_weights(filled: dict) -> None:
    cfg = filled["cfg"]
    h = filled["hists"][("cp", "met_phi")]
    sample = next(s for s in filled["samples"] if s.name == "DY_test")
    df = _reference_frame(sample, cfg)
    in_range = (df.met_phi >= -3.2) & (df.met_phi < 3.2)
    for region, col in [("even", "wt_cp_sm"), ("odd", "wt_cp_ps")]:
        expected = (df.w * df[col])[in_range].sum()
        got = float(
            h[{"process": "DY", "region": region, "variation": "nominal"}].view()["value"].sum()
        )
        assert got == pytest.approx(expected, rel=1e-9), region
    # data sample contributes nothing to cp
    got_data = float(
        h[{"process": "data", "region": "even", "variation": "nominal"}].view()["value"].sum()
    )
    assert got_data == 0.0


def test_fitcp_signal_region_templates(workspace: dict) -> None:
    from wham.config import CPPlotCfg

    cfg = load_config(workspace["yaml"])
    plots = cfg.plots.model_copy(
        update={"fitcp": [CPPlotCfg(var="met_phi", even="wt_cp_sm", odd="wt_cp_ps")]}
    )
    cfg = cfg.model_copy(update={"plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["fitcp"])

    hists: dict = {}
    for s in samples:
        merge_hists(hists, fill_sample(s, skims[s.name], spec))

    h = hists[("fitcp", "met_phi")]
    sample = next(s for s in samples if s.name == "TT_test")
    df = _reference_frame(sample, cfg)
    # SR = trigger & os & iso (abcd) on top of base selection
    sr = (df.trg == 1) & (df.os == 1) & (df.id_2 >= 5)
    in_range = (df.met_phi >= -3.2) & (df.met_phi < 3.2)
    for region, col in [("even", "wt_cp_sm"), ("odd", "wt_cp_ps")]:
        expected = (df.w * df[col])[sr & in_range].sum()
        got = float(
            h[{"process": "TT", "region": region, "variation": "nominal"}].view()["value"].sum()
        )
        assert got == pytest.approx(expected, rel=1e-9), region
    # data and QCD slots stay empty
    for proc in ("data", "QCD"):
        assert float(
            h[{"process": proc, "region": "even", "variation": "nominal"}].view()["value"].sum()
        ) == 0.0


def test_fitcp_cache_key_includes_weight_columns(workspace: dict) -> None:
    from wham.config import CPPlotCfg
    from wham.fill import spec_extras
    from wham.histcache import cache_key

    cfg = load_config(workspace["yaml"])
    plots = cfg.plots.model_copy(
        update={"fitcp": [CPPlotCfg(var="met_phi", even="wt_cp_sm", odd="wt_cp_ps")]}
    )
    cfg = cfg.model_copy(update={"plots": plots})
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=["fitcp"])
    vcfg = cfg.variables["met_phi"].model_dump()

    extras = spec_extras(spec)
    k1 = cache_key(cfg, samples, skims, spec, "fitcp", "met_phi", vcfg,
                   extra=extras[("fitcp", "met_phi")])
    k2 = cache_key(cfg, samples, skims, spec, "fitcp", "met_phi", vcfg,
                   extra={"even": "other_col", "odd": "wt_cp_ps"})
    assert k1 != k2
    # no extra -> matches legacy keying (datamc etc. unaffected)
    k3 = cache_key(cfg, samples, skims, spec, "datamc", "met_phi", vcfg)
    k4 = cache_key(cfg, samples, skims, spec, "datamc", "met_phi", vcfg, extra=None)
    assert k3 == k4


def test_fill_all_uses_cache(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)

    h1 = fill_all(cfg, samples, skims, families=FAMILIES, workers=1, use_cache=True)
    # second call must come purely from cache (cache_only raises otherwise)
    h2 = fill_all(cfg, samples, skims, families=FAMILIES, workers=1,
                  use_cache=True, cache_only=True)
    k = ("datamc", "pt_1")
    assert np.allclose(h1[k].view()["value"], h2[k].view()["value"])


def test_cache_key_sensitivity(workspace: dict) -> None:
    from wham.histcache import cache_key

    cfg = load_config(workspace["yaml"])
    samples, _ = discover_samples(cfg)
    skims = ensure_skims(cfg, samples, workers=1)
    spec = build_fill_spec(cfg, families=FAMILIES)
    vcfg = cfg.variables["pt_1"].model_dump()

    base = cache_key(cfg, samples, skims, spec, "datamc", "pt_1", vcfg)

    cut = cfg.model_copy(update={"selection": "pt_1 > 30"})
    assert cache_key(cut, samples, skims, spec, "datamc", "pt_1", vcfg) != base

    relabeled = dict(vcfg, label="something new")
    assert cache_key(cfg, samples, skims, spec, "datamc", "pt_1", relabeled) == base

    rebinned = dict(vcfg, bins=99)
    assert cache_key(cfg, samples, skims, spec, "datamc", "pt_1", rebinned) != base
