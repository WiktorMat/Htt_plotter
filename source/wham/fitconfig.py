"""Fit configuration: one YAML per fit, referencing an analysis config.

The fitter is generic — `wham fit <config>` and the YAML carries everything:
POIs, how each process's yield depends on them (optionally via weighted
template components), categories (datacard bins), systematics and scans.
The engine knows no physics modes; see Configurations/fits/TEMPLATE.yaml.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from wham import util
from wham.config import AnalysisConfig, ComponentFillCfg, PlotsCfg, _Model, load_config
from wham.expr import ExprError, parse

FITS_DIR_NAME = "fits"

DEFAULT_COMBINE_IMAGE = (
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/combine-standalone:latest"
)


def dc_name(name: str) -> str:
    """Datacard-safe name (combine chokes on '+' etc.)."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


# ------------------------------------------------------------- formulas

FORMULA_FUNCS = ("cos", "sin", "tan", "atan", "sqrt", "exp", "log", "abs", "pow")


def translate_formula(expr: str, pois: list[str]) -> tuple[str, list[str]]:
    """Validate a POI expression and emit a TFormula with positional @i refs.

    Returns (tformula, deps); deps are the referenced POIs in @-index order.
    Powers may be written `^` or `**`; allowed functions: FORMULA_FUNCS.
    """
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid formula {expr!r}: {e.msg}") from None

    deps: list[str] = []
    binops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def emit(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return emit(node.body)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return f"pow({emit(node.left)},{emit(node.right)})"
            op = binops.get(type(node.op))
            if op is None:
                raise ValueError(f"invalid formula {expr!r}: operator not allowed")
            return f"({emit(node.left)}{op}{emit(node.right)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = "-" if isinstance(node.op, ast.USub) else "+"
            return f"({sign}{emit(node.operand)})"
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name)
                    or node.func.id not in FORMULA_FUNCS or node.keywords):
                raise ValueError(
                    f"invalid formula {expr!r}: only functions "
                    f"{', '.join(FORMULA_FUNCS)} are allowed"
                )
            return f"{node.func.id}({','.join(emit(a) for a in node.args)})"
        if isinstance(node, ast.Name):
            if node.id not in pois:
                raise ValueError(
                    f"invalid formula {expr!r}: unknown name '{node.id}' "
                    f"(POIs: {sorted(pois)})"
                )
            if node.id not in deps:
                deps.append(node.id)
            return f"@{deps.index(node.id)}"
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            return repr(node.value)
        raise ValueError(f"invalid formula {expr!r}: unsupported syntax")

    return emit(tree), deps


# ------------------------------------------------------------- schema


class PoiCfg(_Model):
    init: float = 1.0
    range: tuple[float, float]

    @model_validator(mode="after")
    def _ordered(self) -> "PoiCfg":
        lo, hi = self.range
        if not lo < hi:
            raise ValueError(f"POI range must be increasing, got [{lo}, {hi}]")
        if not lo <= self.init <= hi:
            raise ValueError(f"POI init {self.init} outside range [{lo}, {hi}]")
        return self


class ComponentCfg(_Model):
    weight: str  # per-event weight column/expression for this template
    scale: str   # yield expression in the POIs


class ProcessModelCfg(_Model):
    scale: str | None = None                        # single template
    components: dict[str, ComponentCfg] | None = None  # weighted templates

    @model_validator(mode="after")
    def _exactly_one(self) -> "ProcessModelCfg":
        if (self.scale is None) == (self.components is None):
            raise ValueError("a model process needs exactly one of 'scale' or 'components'")
        if self.components is not None and not self.components:
            raise ValueError("'components' must not be empty")
        return self


class ModelCfg(_Model):
    pois: dict[str, PoiCfg]
    processes: dict[str, ProcessModelCfg]

    @model_validator(mode="after")
    def _consistent(self) -> "ModelCfg":
        if not self.pois:
            raise ValueError("model.pois must declare at least one POI")
        if not self.processes:
            raise ValueError("model.processes must scale at least one process")
        pois = list(self.pois)
        used: set[str] = set()
        for proc, pm in self.processes.items():
            scales = ([pm.scale] if pm.scale is not None
                      else [c.scale for c in pm.components.values()])
            for scale in scales:
                try:
                    _, deps = translate_formula(scale, pois)
                except ValueError as e:
                    raise ValueError(f"model.processes.{proc}: {e}") from None
                used.update(deps)
        unused = set(pois) - used
        if unused:
            raise ValueError(f"POIs not used by any scale expression: {sorted(unused)}")
        return self


class CategoryCfg(_Model):
    name: str       # datacard bin name
    variable: str   # observable, a variable of the analysis config
    cut: str | None = None  # extra selection on top of the analysis selection

    @model_validator(mode="after")
    def _name_safe(self) -> "CategoryCfg":
        if not self.name or dc_name(self.name) != self.name:
            raise ValueError(
                f"category name '{self.name}' must be datacard-safe "
                "(letters, digits, underscores)"
            )
        return self


class ScanCfg(_Model):
    pois: list[str] = Field(min_length=1, max_length=2)  # 1 -> 1D, 2 -> 2D grid
    points: int = Field(default=50, gt=1)  # per axis
    range: tuple[float, float] | None = None  # 1D only; default = POI range

    @model_validator(mode="after")
    def _range_1d_only(self) -> "ScanCfg":
        if self.range is not None and len(self.pois) != 1:
            raise ValueError("scan 'range' applies to 1D scans only (2D uses POI ranges)")
        if len(set(self.pois)) != len(self.pois):
            raise ValueError(f"duplicate POI in scan {self.pois}")
        return self


class SystematicCfg(_Model):
    name: str
    effect: Literal["lnN", "rateParam"] = "lnN"
    processes: list[str]  # fnmatch patterns over process/template names
    categories: list[str] | None = None  # restrict to these bins; default all
    scaleFactor: float | None = Field(default=None, gt=0)  # lnN only
    init: float = 1.0  # rateParam starting value
    range: tuple[float, float] | None = None  # rateParam bounds, e.g. [0.1, 5]

    @model_validator(mode="after")
    def _fields_match_effect(self) -> "SystematicCfg":
        if self.effect == "lnN":
            if self.scaleFactor is None:
                raise ValueError(f"systematic '{self.name}': lnN needs a scaleFactor")
            if self.range is not None or self.init != 1.0:
                raise ValueError(f"systematic '{self.name}': init/range are rateParam-only")
        elif self.scaleFactor is not None:
            raise ValueError(f"systematic '{self.name}': rateParam takes no scaleFactor")
        return self


class AsimovCfg(_Model):
    enabled: bool = True  # combine -t -1
    parameters: dict[str, float] = {}  # injected truth; default: POIs at init


class ToyCfg(_Model):
    # Validation knob: modulates the two templates of every 2-component
    # process by (1 +- A*cos(x)) at datacard export, yield-preserving.
    # Every output is labeled TOY when nonzero.
    asymmetry: float = 0.0


class CombineCfg(_Model):
    image: str = DEFAULT_COMBINE_IMAGE


class FitConfig(_Model):
    name: str
    analysis: str  # bare name or path of the analysis YAML
    categories: list[CategoryCfg] = Field(min_length=1)
    model: ModelCfg
    systematics: list[SystematicCfg] = []
    scans: list[ScanCfg] | None = None  # default: one 1D scan per POI
    asimov: AsimovCfg = AsimovCfg()
    toy: ToyCfg = ToyCfg()
    auto_mc_stats: int | None = 10
    combine: CombineCfg = CombineCfg()

    @model_validator(mode="after")
    def _consistent(self) -> "FitConfig":
        names = [c.name for c in self.categories]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate category names: {names}")
        templates = list(scale_map(self).keys())
        if len(set(templates)) != len(templates):
            raise ValueError(f"model template names collide after sanitization: {templates}")
        pois = set(self.model.pois)
        for scan in self.scans or []:
            missing = set(scan.pois) - pois
            if missing:
                raise ValueError(f"scan over undeclared POIs: {sorted(missing)}")
        bad = set(self.asimov.parameters) - pois
        if bad:
            raise ValueError(f"asimov.parameters for undeclared POIs: {sorted(bad)}")
        for syst in self.systematics:
            if syst.categories is not None:
                unknown = set(syst.categories) - set(names)
                if unknown:
                    raise ValueError(
                        f"systematic '{syst.name}' restricted to unknown "
                        f"categories: {sorted(unknown)}"
                    )
        if self.toy.asymmetry != 0.0 and not any(
            pm.components is not None and len(pm.components) == 2
            for pm in self.model.processes.values()
        ):
            raise ValueError("toy.asymmetry needs a process with exactly two components")
        return self

    def resolved_scans(self) -> list[ScanCfg]:
        if self.scans is not None:
            return self.scans
        return [ScanCfg(pois=[p]) for p in self.model.pois]


# ------------------------------------------------------------- derived


def template_names(fit: FitConfig, process: str) -> list[str]:
    """Datacard template names of one model process."""
    pm = fit.model.processes[process]
    if pm.components is not None:
        return [dc_name(f"{process}_{comp}") for comp in pm.components]
    return [dc_name(process)]


def scale_map(fit: FitConfig) -> dict[str, str]:
    """Datacard template name -> yield-scale expression."""
    out: dict[str, str] = {}
    for proc, pm in fit.model.processes.items():
        if pm.components is not None:
            for comp, c in pm.components.items():
                out[dc_name(f"{proc}_{comp}")] = c.scale
        else:
            out[dc_name(proc)] = pm.scale
    return out


def datacard_processes(fit: FitConfig, cfg: AnalysisConfig) -> tuple[list[str], list[str]]:
    """(model template names, background names) in stack order; model
    templates are the datacard 'signals' (ids <= 0)."""
    signals: list[str] = []
    backgrounds: list[str] = []
    for proc in cfg.stack_order():
        if proc in fit.model.processes:
            signals += template_names(fit, proc)
        else:
            backgrounds.append(dc_name(proc))
    return signals, backgrounds


def template_parents(fit: FitConfig, cfg: AnalysisConfig) -> dict[str, str]:
    """Datacard name -> config process name, for systematic pattern matching
    (covers both component templates and sanitized names like W+jets)."""
    parents = {dc_name(p): p for p in cfg.stack_order()}
    for proc in fit.model.processes:
        for name in template_names(fit, proc):
            parents[name] = proc
    return parents


def syst_matches(patterns: list[str], dc: str, parent: str | None = None) -> bool:
    """A systematic pattern matches the datacard name or its parent process."""
    for pattern in patterns:
        if fnmatch.fnmatch(dc, pattern):
            return True
        if parent is not None and fnmatch.fnmatch(parent, pattern):
            return True
    return False


def fit_families(fit: FitConfig) -> list[str]:
    has_components = any(pm.components for pm in fit.model.processes.values())
    return ["datamc"] + (["fitcp"] if has_components else [])


def _component_fills(fit: FitConfig, variable: str) -> list[ComponentFillCfg]:
    return [
        ComponentFillCfg(
            var=variable, process=proc,
            components={name: c.weight for name, c in pm.components.items()},
        )
        for proc, pm in fit.model.processes.items()
        if pm.components is not None
    ]


def category_analysis(fit: FitConfig, cfg: AnalysisConfig, cat: CategoryCfg) -> AnalysisConfig:
    """Analysis config for one category: cut folded into the selection,
    plots reduced to exactly the fills this category's templates need."""
    selection = cfg.selection if cat.cut is None else f"({cfg.selection}) & ({cat.cut})"
    plots = PlotsCfg(datamc=[cat.variable], fitcp=_component_fills(fit, cat.variable))
    return cfg.model_copy(update={"selection": selection, "plots": plots})


def union_analysis(fit: FitConfig, cfg: AnalysisConfig) -> AnalysisConfig:
    """One config whose required_columns() covers every category fill —
    drives the skim build once. Its selection is never evaluated."""
    cuts = [c.cut for c in fit.categories if c.cut]
    selection = " & ".join(f"({s})" for s in [cfg.selection, *cuts])
    variables = list(dict.fromkeys(c.variable for c in fit.categories))
    fitcp = [f for v in variables for f in _component_fills(fit, v)]
    plots = PlotsCfg(datamc=variables, fitcp=fitcp)
    return cfg.model_copy(update={"selection": selection, "plots": plots})


# ------------------------------------------------------------- loading


def resolve_fit_config(config: str | None) -> Path:
    """Accept a path, a bare name (Configurations/fits/<name>.yaml), or
    nothing when exactly one fit YAML exists."""
    fits_dir = util.repo_root() / "Configurations" / FITS_DIR_NAME

    if config is None:
        candidates = [p for p in sorted(fits_dir.glob("*.yaml")) if p.stem != "TEMPLATE"]
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
    available = ", ".join(p.stem for p in sorted(fits_dir.glob("*.yaml"))) or "<none>"
    raise FileNotFoundError(
        f"no fit config '{config}' and no {candidate} (available: {available})"
    )


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


def load_fit_config(path: str | Path) -> tuple[FitConfig, AnalysisConfig]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    fit = FitConfig.model_validate(raw)

    cfg = load_config(_resolve_analysis(fit.analysis, path.parent))

    # ---- cross-validation against the analysis config
    for cat in fit.categories:
        if cat.variable not in cfg.variables:
            raise ValueError(
                f"category '{cat.name}': variable '{cat.variable}' is not defined "
                f"in '{fit.analysis}' ({sorted(cfg.variables)})"
            )
        if cat.cut is not None:
            try:
                parse(cat.cut)
            except ExprError as e:
                raise ValueError(f"category '{cat.name}': invalid cut: {e}") from None

    for proc, pm in fit.model.processes.items():
        pcfg = cfg.processes.get(proc)
        if pcfg is None:
            raise ValueError(
                f"model process '{proc}' is not a process of '{fit.analysis}' "
                f"({sorted(cfg.processes)})"
            )
        if pcfg.kind == "data":
            raise ValueError(f"model process '{proc}' is the data process")
        if pm.components is not None:
            if pcfg.kind != "mc":
                raise ValueError(
                    f"model process '{proc}': components need per-event weights, "
                    f"so only kind=mc processes can be split (got kind={pcfg.kind})"
                )
            for comp, c in pm.components.items():
                try:
                    parse(c.weight)
                except ExprError as e:
                    raise ValueError(
                        f"model process '{proc}', component '{comp}': "
                        f"invalid weight: {e}"
                    ) from None

    if cfg.data_process() is None:
        raise ValueError(f"analysis '{fit.analysis}' has no kind=data process")

    signals, backgrounds = datacard_processes(fit, cfg)
    parents = template_parents(fit, cfg)
    all_dc = signals + backgrounds
    for syst in fit.systematics:
        if not any(syst_matches(syst.processes, n, parents.get(n)) for n in all_dc):
            raise ValueError(
                f"systematic '{syst.name}' matches no datacard process "
                f"(patterns {syst.processes}, processes {all_dc})"
            )

    return fit, cfg
