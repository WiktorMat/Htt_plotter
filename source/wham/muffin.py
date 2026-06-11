"""muffin: BDT fake-factor scores baked into the skims at plotting level.

Adapted from higgs-dna-waw scripts/ditau/post_processing/add_bdtfakefactorscores.py.

When the analysis YAML carries a `fake_factors:` block, the skim build applies
the XGBoost BDT FF models to the hadronic tau leg and appends
BDT_FF_score_<process>_sublead columns (plus _<Unc>_up/down with
systematics: true). The original EOS inputs are never touched; the scores live
only in the local skim cache and behave like any other column afterwards:
plot them as variables or point qcd.ff_weight at one.

Skims are keyed on the model files, so swapping or retraining a model rebuilds
them automatically.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

from wham.util import file_signature

# Era labels the BDT FF models were trained with. Run3_2024 is not in the
# training set yet; set era_label explicitly to borrow the closest trained
# era until the models are retrained.
ERA_LABELS = {
    "Run3_2022": 0,
    "Run3_2022EE": 1,
    "Run3_2023": 2,
    "Run3_2023BPix": 3,
}

# model feature -> candidate source columns for the scored (tau_h) leg,
# tried in order. Engineered features (jpt_pt, met_var_qcd, ...) build on these.
SUBLEAD_SOURCES = {
    "decayMode": ("decayMode_2",),
    "decayModePNet": ("decayModePNet_2",),
    "pt": ("pt_2",),
    "eta": ("eta_2",),
    "phi": ("phi_2",),
    "charge": ("charge_2",),
    "jpt": ("seeding_jpt_2", "jpt_2"),
    "n_jets": ("n_jets",),
    "n_bjets": ("n_bjets",),
    "met_pt": ("met_pt",),
    "met_dphi": ("met_dphi_2",),
}

# Uncertainty maps lifted verbatim from the higgs-dna script (mt entries).
NONCLOSURE = {"QCD": 0.05, "Wjets": 0.05, "WjetsMC": 0.05, "ttbarMC": 0.05}
MODELLING = {"QCD": 0.02, "Wjets": 0.01, "WjetsMC": 0.02, "ttbarMC": 0.005}
EXTRAPOLATION = {
    "QCD": [(0, 75, 0.10), (75, float("inf"), 0.20)],
    "Wjets": [(0, float("inf"), 0.05)],
    "WjetsMC": [(0, 75, 0.05), (75, float("inf"), 0.10)],
    "ttbarMC": [(0, float("inf"), 0.02)],
}
BKGSUB_VARIATION = {"QCD": 0.20, "Wjets": 0.10}


def source_columns() -> frozenset[str]:
    """Every input column the scoring might read."""
    return frozenset(c for sources in SUBLEAD_SOURCES.values() for c in sources)


def model_file(models_dir: Path, channel: str, process: str) -> Path:
    """Locate the model, supporting both layouts:
    higgs-dna   <models>/model_<channel>_<process>/model.json
    trainings   <models>/<channel>_<process>/best_model.json
    """
    candidates = [
        Path(models_dir) / d / f
        for d in (f"model_{channel}_{process}", f"{channel}_{process}")
        for f in ("model.json", "best_model.json")
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no model for {channel}_{process}; looked for " +
        ", ".join(str(p) for p in candidates)
    )


def score_column(process: str) -> str:
    return f"BDT_FF_score_{process}_sublead"


def signature(ff: Any) -> dict | None:
    """Cache payload identifying the models; changes when a model is retrained."""
    if ff is None:
        return None
    sig: dict[str, Any] = {
        "channel": ff.channel,
        "era_label": ff.resolved_era_label(),
        "processes": sorted(ff.processes),
        "systematics": ff.systematics,
        "models": {},
    }
    for process in ff.processes:
        mfile = model_file(ff.models, ff.channel, process)
        entry = {"model": file_signature(mfile)}
        tpath = mfile.parent / "temperature_scaling_results.json"
        if tpath.is_file():
            entry["temperature"] = file_signature(tpath)
        sig["models"][process] = entry
    return sig


# ------------------------------------------------------------- scoring math


def mask_denominator(probabilities, epsilon=1e-6):
    """Zero the subtraction classes where the FF denominator is ill-conditioned."""
    masked = probabilities.copy()
    ill = (masked[:, 1] - masked[:, 3]) <= epsilon
    masked[ill, 2] = 0.0
    masked[ill, 3] = 0.0
    return masked


def renormalise_probabilities(probabilities):
    import numpy as np

    s = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.where(s == 0, 1, s)


def multiclass_ff(probabilities):
    """FF = (p_dataAR - p_mcAR) / (p_dataSR - p_mcSR), with a no-subtraction fallback.

    Returns (score, raw_ff, masked_probabilities); raw_ff keeps the sign that
    triggers the fallback, which the BkgSub variation needs to reproduce.
    """
    import numpy as np

    p = renormalise_probabilities(mask_denominator(probabilities))
    raw = (p[:, 0] - p[:, 2]) / (p[:, 1] - p[:, 3])
    return np.where(raw < 0, p[:, 0] / p[:, 1], raw), raw, p


def apply_temperature(probabilities, temperature: float):
    import numpy as np

    if temperature == 1.0:
        return probabilities
    if probabilities.ndim == 2:  # multiclass softprob
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        logits -= logits.mean(axis=1, keepdims=True)
        return np.exp(logits / temperature)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0) / np.clip(1 - probabilities, 1e-12, 1.0))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def bkgsub_scores(probabilities, variation: float):
    """FF recomputed with the MC-subtraction classes scaled by +-variation."""
    import numpy as np

    _, raw, p = multiclass_ff(probabilities)
    out = {}
    for sign, key in ((1, "down"), (-1, "up")):  # more subtraction -> smaller FF
        scaled = probabilities.copy()
        scaled[:, 2] *= 1 + sign * variation
        scaled[:, 3] *= 1 + sign * variation
        ff, _, _ = multiclass_ff(scaled)
        out[key] = np.where(raw < 0, p[:, 0] / p[:, 1], ff)
    return out["up"], out["down"]


def extrapolation_uncertainty(process: str, pt):
    import numpy as np

    unc = np.zeros(len(pt), dtype=float)
    for low, high, val in EXTRAPOLATION.get(process, []):
        unc[(pt >= low) & (pt < high)] = val
    return unc


# ------------------------------------------------------------- models


class FFModel:
    """One process's model + calibration, applied to feature frames."""

    def __init__(self, models_dir: Path, channel: str, process: str, want_bootstrap: bool):
        import xgboost as xgb

        self.process = process
        mfile = model_file(models_dir, channel, process)
        mdir = mfile.parent
        self.booster = xgb.Booster()
        self.booster.load_model(str(mfile))
        self.features = self.booster.feature_names
        if not self.features:
            raise RuntimeError(f"{mfile} carries no feature names")
        self.multiclass = "multi" in json.loads(
            self.booster.save_config())["learner"]["objective"]["name"]

        self.temperature = 1.0
        tpath = mdir / "temperature_scaling_results.json"
        if tpath.is_file():
            with open(tpath) as f:
                self.temperature = float(json.load(f).get("optimal_temperature", 1.0))

        self.bootstrap: list = []
        if want_bootstrap:
            self.bootstrap = self._load_bootstrap(
                mdir / "bootstrap_models" / "bootstrap_models.tar.gz")

    def _load_bootstrap(self, tar_path: Path) -> list:
        import xgboost as xgb

        # silently absent or a git-lfs pointer stub -> no Bootstrap columns
        if not tar_path.is_file() or not tarfile.is_tarfile(tar_path):
            return []
        out = []
        with tarfile.open(tar_path, "r:gz") as tar:
            for mem in sorted(tar.getmembers(), key=lambda m: m.name):
                if not (mem.isfile() and mem.name.endswith(".json")):
                    continue
                fobj = tar.extractfile(mem)
                if fobj is None:
                    continue
                model = xgb.Booster()
                model.load_model(bytearray(fobj.read()))
                out.append(model)
        return out

    def _ff_from_pred(self, pred):
        if self.multiclass:
            return multiclass_ff(pred)[0]
        return (1.0 - pred) / pred

    def score(self, frame, systematics: bool) -> dict:
        """{column name: np.ndarray} for one engineered feature frame."""
        import numpy as np
        import xgboost as xgb

        dmat = xgb.DMatrix(frame[self.features])
        pred = apply_temperature(self.booster.predict(dmat), self.temperature)
        score = self._ff_from_pred(pred)
        tag = score_column(self.process)
        out = {tag: score}
        if not systematics:
            return out

        if self.multiclass:
            up, down = bkgsub_scores(pred, BKGSUB_VARIATION[self.process])
        else:
            up = down = np.zeros_like(score)
        out[f"{tag}_BkgSub_up"], out[f"{tag}_BkgSub_down"] = up, down

        mod = MODELLING.get(self.process, 0.0)
        out[f"{tag}_Modelling_up"] = score * (1 + mod)
        out[f"{tag}_Modelling_down"] = score * (1 - mod)

        ext = extrapolation_uncertainty(self.process, np.asarray(frame["pt"]))
        out[f"{tag}_Extrapolation_up"] = score * (1 + ext)
        out[f"{tag}_Extrapolation_down"] = score * (1 - ext)

        nc = NONCLOSURE.get(self.process, 0.0)
        out[f"{tag}_NonClosure_up"] = score * (1 + nc)
        out[f"{tag}_NonClosure_down"] = score * (1 - nc)

        if self.bootstrap:
            preds = np.stack([self._ff_from_pred(m.predict(dmat)) for m in self.bootstrap])
            sigma = preds.std(axis=0)
            out[f"{tag}_Bootstrap_up"] = score + sigma
            out[f"{tag}_Bootstrap_down"] = score - sigma
        return out


def build_features(table, era_label: int):
    """Engineered tau_h-leg feature frame for a pyarrow table."""
    import numpy as np
    import pandas as pd

    cols = {}
    missing = []
    for feat, sources in SUBLEAD_SOURCES.items():
        src = next((s for s in sources if s in table.column_names), None)
        if src is None:
            missing.append(f"{feat} (looked for {', '.join(sources)})")
            continue
        cols[feat] = table.column(src).to_numpy(zero_copy_only=False)
    if missing:
        raise RuntimeError(
            "fake_factors: input is missing model feature columns: " + "; ".join(missing)
        )

    frame = pd.DataFrame(cols)
    frame["jpt_pt"] = (frame["jpt"] / frame["pt"]).clip(lower=0)
    frame["met_var_qcd"] = (frame["met_pt"] / frame["pt"]) * np.cos(frame["met_dphi"])
    frame["era_label"] = era_label
    frame["is_lead_tau"] = 0
    return frame


def augment_table(ff: Any, table):
    """Append the BDT FF score columns to a pyarrow table."""
    import pyarrow as pa

    frame = build_features(table, ff.resolved_era_label())
    for process in ff.processes:
        model = FFModel(ff.models, ff.channel, process, want_bootstrap=ff.systematics)
        for name, values in model.score(frame, ff.systematics).items():
            table = table.append_column(name, pa.array(values, type=pa.float64()))
    return table
