from __future__ import annotations

import pytest

from wham.config import discover_samples, load_config, sample_scale


def test_load_and_discover(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    assert cfg.name == "test_analysis"
    assert cfg.stack_order() == ["QCD", "TT", "DY"]
    assert cfg.data_process() == "data"
    assert cfg.qcd_process() == "QCD"

    samples, warnings = discover_samples(cfg)
    assert [s.name for s in samples] == ["DY_test", "Muon_test", "TT_test"]
    assert not warnings
    by_name = {s.name: s for s in samples}
    assert by_name["Muon_test"].kind == "data"
    assert sample_scale(by_name["Muon_test"], cfg.lumi) == 1.0
    # lumi * xs * filter_eff / eff = 1000 * 200 * 0.5 / 80000
    assert sample_scale(by_name["DY_test"], cfg.lumi) == pytest.approx(1.25)


def test_required_columns(workspace: dict) -> None:
    cfg = load_config(workspace["yaml"])
    cols = cfg.required_columns()
    assert {"pt_1", "pt_2", "eta_1", "m_vis", "met_phi", "os", "id_2",
            "trg", "weight", "wt_cp_sm", "wt_cp_ps"} <= cols


@pytest.mark.parametrize(
    "patch, match",
    [
        ("selection: 'pt_1 >'", "invalid expression in 'selection'"),
        ("selection: 'sin(pt_1) > 1'", "only abs"),
        ("plots:\n  datamc: [nonexistent_var]", "not defined in 'variables'"),
        ("lumi: -5", "greater than 0"),
        ("typo_field: 1", "extra"),
    ],
)
def test_config_rejections(workspace: dict, patch: str, match: str) -> None:
    yaml_path = workspace["yaml"]
    base = yaml_path.read_text()
    bad = workspace["tmp"] / "bad.yaml"
    if patch.startswith(("selection:", "lumi:")):
        key = patch.split(":")[0]
        lines = [l for l in base.splitlines() if not l.startswith(f"{key}:")]
        bad.write_text("\n".join(lines) + f"\n{patch}\n")
    elif patch.startswith("plots:"):
        head = base.split("plots:")[0]
        bad.write_text(head + patch + "\n")
    else:
        bad.write_text(base + f"\n{patch}\n")
    with pytest.raises(Exception, match=match):
        load_config(bad)


def test_ambiguous_pattern_rejected(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    bad = workspace["tmp"] / "ambig.yaml"
    bad.write_text(base.replace('TT:   {samples: ["TT_*"]', 'TT:   {samples: ["*_test"]'))
    cfg = load_config(bad)
    with pytest.raises(ValueError, match="matches both"):
        discover_samples(cfg)


def test_unmatched_pattern_warns(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    cfg_path = workspace["tmp"] / "warn.yaml"
    cfg_path.write_text(base.replace('["DY_*"]', '["DY_*", "Zprime_*"]'))
    _, warnings = discover_samples(load_config(cfg_path))
    assert any("Zprime_*" in w for w in warnings)


def test_variable_binning_validation() -> None:
    from wham.config import VariableCfg

    VariableCfg(bins=10, range=(0, 1))            # count + range
    v = VariableCfg(bins=[0, 1, 5, 20])           # explicit edges
    assert v.span() == (0, 20)
    with pytest.raises(ValueError, match="needs an explicit 'range'"):
        VariableCfg(bins=10)
    with pytest.raises(ValueError, match="must be omitted"):
        VariableCfg(bins=[0, 1], range=(0, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        VariableCfg(bins=[0, 5, 5, 10])
    with pytest.raises(ValueError, match="at least 2"):
        VariableCfg(bins=[3])
    with pytest.raises(ValueError, match="increasing"):
        VariableCfg(bins=5, range=(1, 0))
    with pytest.raises(ValueError, match="plain column name"):
        VariableCfg(bins=5, range=(0, 1), column="pt_1 + pt_2")


def test_column_alias_resolution(workspace: dict) -> None:
    from wham.config import VariableCfg

    cfg = load_config(workspace["yaml"])
    variables = dict(cfg.variables)
    variables["m_vis_coarse"] = VariableCfg(column="m_vis", bins=[0.0, 50.0, 250.0])
    plots = cfg.plots.model_copy(update={"datamc": ["m_vis_coarse"]})
    cfg = cfg.model_copy(update={"variables": variables, "plots": plots})

    assert cfg.column_of("m_vis_coarse") == "m_vis"
    assert cfg.column_of("pt_1") == "pt_1"
    cols = cfg.required_columns()
    assert "m_vis" in cols
    assert "m_vis_coarse" not in cols  # aliases are not real file columns


def test_params_side_file(workspace: dict) -> None:
    base = workspace["yaml"].read_text()
    lines = base.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("sample_params:"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
    side_dir = workspace["tmp"] / "side"
    side_dir.mkdir()
    (side_dir / "analysis.yaml").write_text("\n".join(lines[:start] + lines[end:]))
    (side_dir / "params.yaml").write_text(
        "lumi: 999\nTT_test:\n  xs: 100.0\n  eff: 50000\nDY_test:\n  xs: 200.0\n  eff: 80000\n"
    )
    cfg = load_config(side_dir / "analysis.yaml")
    assert cfg.lumi == 1000.0  # analysis yaml wins, side-file lumi ignored
    assert cfg.sample_params["TT_test"].xs == 100.0
