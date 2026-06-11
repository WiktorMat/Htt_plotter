"""Fit configuration: one YAML per fit, referencing an analysis config."""

from __future__ import annotations

import fnmatch
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from wham import util
from wham.config import AnalysisConfig, _Model, load_config

FITS_DIR_NAME = "fits"

DEFAULT_COMBINE_IMAGE = (
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/combine-standalone:latest"
)


class SystematicCfg(_Model):
    name: str
    effect: Literal["lnN"] = "lnN"
    processes: list[str]  # fnmatch patterns over datacard process names
    scaleFactor: float = Field(gt=0)


class ScanCfg(_Model):
    points: int = Field(default=50, gt=1)
    range: tuple[float, float] | None = None  # default depends on the POI


class AsimovCfg(_Model):
    enabled: bool = True  # combine -t -1
    parameters: dict[str, float] = {}  # injected truth, e.g. {alpha: 0.4}


class ToyCfg(_Model):
    # Artificial +-A*cos(x) modulation of the signal even/odd templates,
    # applied at datacard export only. For exercising the machinery; every
    # output is labeled TOY when nonzero.
    asymmetry: float = 0.0


class CombineCfg(_Model):
    image: str = DEFAULT_COMBINE_IMAGE


class FitConfig(_Model):
    name: str
    analysis: str  # bare name or path of the analysis YAML
    variable: str
    mode: Literal["rate", "cp"]
    signal: str
    bin: str = "mt_SR"
    even: str = "wt_cp_sm"
    odd: str = "wt_cp_ps"
    auto_mc_stats: int | None = 10
    systematics: list[SystematicCfg] = []
    scan: ScanCfg = ScanCfg()
    asimov: AsimovCfg = AsimovCfg()
    toy: ToyCfg = ToyCfg()
    combine: CombineCfg = CombineCfg()

    def poi(self) -> str:
        return "alpha" if self.mode == "cp" else "r"

    def scan_range(self) -> tuple[float, float]:
        if self.scan.range is not None:
            return self.scan.range
        return (0.0, math.pi / 2) if self.mode == "cp" else (0.0, 3.0)

    @model_validator(mode="after")
    def _toy_needs_cp(self) -> "FitConfig":
        if self.toy.asymmetry != 0.0 and self.mode != "cp":
            raise ValueError("toy.asymmetry only applies to mode: cp")
        return self


def resolve_fit_config(config: str | None) -> Path:
    """Accept a path, a bare name (Configurations/fits/<name>.yaml), or
    nothing when exactly one fit YAML exists."""
    fits_dir = util.repo_root() / "Configurations" / FITS_DIR_NAME

    if config is None:
        candidates = sorted(fits_dir.glob("*.yaml"))
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(p.stem for p in candidates) or "<none>"
        raise FileNotFoundError(
            f"choose a fit config: found {len(candidates)} in {fits_dir} ({names})"
        )

    path = Path(config)
    if path.is_file():
        return path
    candidate = fits_dir / f"{config}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no fit config '{config}' and no {candidate}")


def _resolve_analysis(ref: str, fit_yaml_dir: Path) -> Path:
    path = Path(ref)
    for candidate in (
        path,
        fit_yaml_dir / path,
        util.repo_root() / "Configurations" / f"{ref}.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"analysis config '{ref}' not found")


def datacard_processes(fit: FitConfig, cfg: AnalysisConfig) -> tuple[list[str], list[str]]:
    """(signal_names, background_names) as they appear in the datacard,
    in stack order. cp mode splits the signal into even/odd templates."""
    if fit.mode == "cp":
        signals = [f"{fit.signal}_cpeven", f"{fit.signal}_cpodd"]
    else:
        signals = [fit.signal]
    backgrounds = [p for p in cfg.stack_order() if p != fit.signal]
    return signals, backgrounds


def syst_matches(patterns: list[str], dc_name: str, parent: str | None = None) -> bool:
    """A systematic pattern matches the datacard name or its parent process."""
    for pattern in patterns:
        if fnmatch.fnmatch(dc_name, pattern):
            return True
        if parent is not None and fnmatch.fnmatch(parent, pattern):
            return True
    return False


def augmented_analysis(fit: FitConfig, cfg: AnalysisConfig) -> tuple[AnalysisConfig, list[str]]:
    """Analysis config copy with the fit fills wired in, plus the fill families.

    Ensures the fit variable is filled in the datamc family (data_obs, QCD,
    backgrounds) and, in cp mode, as even/odd hypothesis templates (fitcp).
    """
    from wham.config import CPPlotCfg

    datamc = list(dict.fromkeys([*cfg.plots.datamc, fit.variable]))
    fitcp = (
        [CPPlotCfg(var=fit.variable, even=fit.even, odd=fit.odd)]
        if fit.mode == "cp"
        else []
    )
    plots = cfg.plots.model_copy(update={"datamc": datamc, "fitcp": fitcp})
    families = ["datamc"] + (["fitcp"] if fit.mode == "cp" else [])
    return cfg.model_copy(update={"plots": plots}), families


def load_fit_config(path: str | Path) -> tuple[FitConfig, AnalysisConfig]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    fit = FitConfig.model_validate(raw)

    cfg = load_config(_resolve_analysis(fit.analysis, path.parent))

    # ---- cross-validation against the analysis config
    if fit.variable not in cfg.variables:
        raise ValueError(
            f"fit variable '{fit.variable}' is not defined in "
            f"'{fit.analysis}' variables ({sorted(cfg.variables)})"
        )
    signal_cfg = cfg.processes.get(fit.signal)
    if signal_cfg is None or signal_cfg.kind != "mc":
        raise ValueError(
            f"signal '{fit.signal}' must be a kind=mc process of '{fit.analysis}' "
            f"(mc processes: {[n for n, p in cfg.processes.items() if p.kind == 'mc']})"
        )
    if cfg.data_process() is None:
        raise ValueError(f"analysis '{fit.analysis}' has no kind=data process")

    signals, backgrounds = datacard_processes(fit, cfg)
    parents = {f"{fit.signal}_cpeven": fit.signal, f"{fit.signal}_cpodd": fit.signal}
    all_dc = signals + backgrounds
    for syst in fit.systematics:
        if not any(syst_matches(syst.processes, n, parents.get(n)) for n in all_dc):
            raise ValueError(
                f"systematic '{syst.name}' matches no datacard process "
                f"(patterns {syst.processes}, processes {all_dc})"
            )

    return fit, cfg
