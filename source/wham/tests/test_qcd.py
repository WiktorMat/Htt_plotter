from __future__ import annotations

import hist
import numpy as np
import pytest

from wham.config import load_config
from wham.qcd import estimate_qcd

PROCESSES = ["QCD", "TT", "DY", "data"]
REGIONS = ["OS_iso", "SS_iso", "OS_antiiso", "SS_antiiso"]


def _hist(regions=REGIONS, nbins=3):
    return hist.Hist(
        hist.axis.StrCategory(PROCESSES, name="process"),
        hist.axis.StrCategory(regions, name="region"),
        hist.axis.StrCategory(["nominal"], name="variation"),
        hist.axis.Regular(nbins, 0, 3, name="x"),
        storage=hist.storage.Weight(),
    )


def _set(h, proc, region, counts, sumw2=None):
    ip = PROCESSES.index(proc)
    ir = list(h.axes["region"]).index(region)
    h.view()[ip, ir, 0, :]["value"] = counts
    h.view()[ip, ir, 0, :]["variance"] = sumw2 if sumw2 is not None else counts


def _get(h, proc, region):
    view = h[{"process": proc, "region": region, "variation": "nominal"}].view()
    return view["value"], view["variance"]


def test_abcd_transfer_factor(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])  # qcd.method = abcd
    h = _hist()

    _set(h, "data", "SS_iso", [100.0, 50.0, 10.0])
    _set(h, "TT", "SS_iso", [40.0, 20.0, 15.0])     # bin3: mc > data -> clip to 0
    _set(h, "data", "OS_antiiso", [30.0, 30.0, 30.0])
    _set(h, "TT", "OS_antiiso", [10.0, 10.0, 10.0])  # qcd_os_anti = 20
    _set(h, "data", "SS_antiiso", [10.0, 0.0, 10.0])  # bin2: denominator 0 -> tf 0
    _set(h, "TT", "SS_antiiso", [0.0, 0.0, 0.0])

    estimate_qcd(cfg, {("datamc", "x"): h})

    # tf = [20/10, 0 (safe div), 20/10] = [2, 0, 2]
    # ss_iso qcd = [60, 30, 0] -> os_iso qcd = [120, 0, 0]
    counts, variance = _get(h, "QCD", "OS_iso")
    assert np.allclose(counts, [120.0, 0.0, 0.0])
    # sumw2_ss_iso = data + mc poisson = 100+40, 50+20, 10+15 -> * tf^2
    assert np.allclose(variance, [140 * 4.0, 0.0, 25 * 4.0])

    ss_counts, ss_var = _get(h, "QCD", "SS_iso")
    assert np.allclose(ss_counts, [60.0, 30.0, 0.0])
    assert np.allclose(ss_var, [140.0, 70.0, 25.0])


def test_ss_method(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    cfg = cfg.model_copy(
        update={"qcd": cfg.qcd.model_copy(update={"method": "ss", "ff": 1.5})}
    )
    h = _hist(regions=["OS", "SS"])
    _set(h, "data", "SS", [100.0, 10.0, 5.0])
    _set(h, "TT", "SS", [40.0, 20.0, 5.0])  # bin2 clips to 0

    estimate_qcd(cfg, {("datamc", "x"): h})

    counts, variance = _get(h, "QCD", "OS")
    assert np.allclose(counts, [90.0, 0.0, 0.0])
    assert np.allclose(variance, np.array([140.0, 30.0, 10.0]) * 1.5**2)


def test_qcd_noop_without_qcd_process(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    no_qcd = {n: p for n, p in cfg.processes.items() if p.kind != "qcd"}
    cfg = cfg.model_copy(update={"processes": no_qcd})
    h = _hist()
    _set(h, "data", "SS_iso", [10.0, 10.0, 10.0])
    before = h.view().copy()
    estimate_qcd(cfg, {("datamc", "x"): h})
    assert np.array_equal(h.view(), before)
