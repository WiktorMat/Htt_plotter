from __future__ import annotations

from collections.abc import Callable

import pandas as pd


SELECTION_COLUMNS = [
    "os",
    "pt_1",
    "pt_2",
    "eta_1",
    "eta_2",
    "decayModePNet_2",
    "idDeepTau2018v2p5VSjet_2",
    "idDeepTau2018v2p5VSe_2",
    "idDeepTau2018v2p5VSmu_2",
    "iso_1",
    "ip_LengthSig_1",
]


def selection_columns_used(selection_cfg: dict) -> list[str]:
    used: set[str] = set()

    key_to_col = {
        "pt_1_min": "pt_1",
        "pt_2_min": "pt_2",
        "eta_1_abs_max": "eta_1",
        "eta_2_abs_max": "eta_2",
        "decayModePNet_2_eq": "decayModePNet_2",
        "idDeepTau2018v2p5VSjet_2_min": "idDeepTau2018v2p5VSjet_2",
        "idDeepTau2018v2p5VSe_2_min": "idDeepTau2018v2p5VSe_2",
        "idDeepTau2018v2p5VSmu_2_min": "idDeepTau2018v2p5VSmu_2",
        "iso_1_max": "iso_1",
        "ip_LengthSig_1_abs_min": "ip_LengthSig_1",
    }

    for k, col in key_to_col.items():
        if k in (selection_cfg or {}):
            used.add(col)

    return sorted(used)


def selection_mask(df: pd.DataFrame, selection_cfg: dict) -> pd.Series:
    conditions = []

    if "pt_1" in df.columns and "pt_1_min" in selection_cfg:
        conditions.append(df["pt_1"] > float(selection_cfg["pt_1_min"]))

    if "pt_2" in df.columns and "pt_2_min" in selection_cfg:
        conditions.append(df["pt_2"] > float(selection_cfg["pt_2_min"]))

    if "eta_1" in df.columns and "eta_1_abs_max" in selection_cfg:
        conditions.append(df["eta_1"].abs() < float(selection_cfg["eta_1_abs_max"]))

    if "eta_2" in df.columns and "eta_2_abs_max" in selection_cfg:
        conditions.append(df["eta_2"].abs() < float(selection_cfg["eta_2_abs_max"]))

    if "decayModePNet_2" in df.columns and "decayModePNet_2_eq" in selection_cfg:
        conditions.append(df["decayModePNet_2"] == int(selection_cfg["decayModePNet_2_eq"]))

    if "idDeepTau2018v2p5VSjet_2" in df.columns and "idDeepTau2018v2p5VSjet_2_min" in selection_cfg:
        conditions.append(
            df["idDeepTau2018v2p5VSjet_2"] >= int(selection_cfg["idDeepTau2018v2p5VSjet_2_min"])
        )

    if "idDeepTau2018v2p5VSe_2" in df.columns and "idDeepTau2018v2p5VSe_2_min" in selection_cfg:
        conditions.append(
            df["idDeepTau2018v2p5VSe_2"] >= int(selection_cfg["idDeepTau2018v2p5VSe_2_min"])
        )

    if "idDeepTau2018v2p5VSmu_2" in df.columns and "idDeepTau2018v2p5VSmu_2_min" in selection_cfg:
        conditions.append(
            df["idDeepTau2018v2p5VSmu_2"] >= int(selection_cfg["idDeepTau2018v2p5VSmu_2_min"])
        )

    if "iso_1" in df.columns and "iso_1_max" in selection_cfg:
        conditions.append(df["iso_1"] < float(selection_cfg["iso_1_max"]))

    if "ip_LengthSig_1" in df.columns and "ip_LengthSig_1_abs_min" in selection_cfg:
        conditions.append(df["ip_LengthSig_1"].abs() > float(selection_cfg["ip_LengthSig_1_abs_min"]))

    if not conditions:
        return pd.Series(True, index=df.index)

    mask = conditions[0].astype(bool)
    for cond in conditions[1:]:
        mask &= cond.astype(bool)

    return mask


def make_selector(plotter_config: dict) -> Callable[[pd.DataFrame], pd.DataFrame]:
    selection_cfg = (plotter_config or {}).get("selection", {})

    def _selector(df: pd.DataFrame) -> pd.DataFrame:
        return df[selection_mask(df, selection_cfg)]

    return _selector


def make_arrow_filter(plotter_config: dict, available_columns: set[str] | None = None):
    """Build a pyarrow.dataset filter expression from plotter.yaml selection.

    Returns None if no filter can be built.
    """

    selection_cfg = (plotter_config or {}).get("selection", {})
    if not selection_cfg:
        return None

    try:
        import pyarrow.dataset as ds
    except Exception:
        return None

    available_columns = available_columns or set()

    expr = None

    def _and(e):
        nonlocal expr
        expr = e if expr is None else (expr & e)

    if "pt_1_min" in selection_cfg and (not available_columns or "pt_1" in available_columns):
        _and(ds.field("pt_1") > float(selection_cfg["pt_1_min"]))

    if "pt_2_min" in selection_cfg and (not available_columns or "pt_2" in available_columns):
        _and(ds.field("pt_2") > float(selection_cfg["pt_2_min"]))

    if "eta_1_abs_max" in selection_cfg and (not available_columns or "eta_1" in available_columns):
        m = float(selection_cfg["eta_1_abs_max"])
        _and((ds.field("eta_1") < m) & (ds.field("eta_1") > -m))

    if "eta_2_abs_max" in selection_cfg and (not available_columns or "eta_2" in available_columns):
        m = float(selection_cfg["eta_2_abs_max"])
        _and((ds.field("eta_2") < m) & (ds.field("eta_2") > -m))

    if "decayModePNet_2_eq" in selection_cfg and (
        not available_columns or "decayModePNet_2" in available_columns
    ):
        _and(ds.field("decayModePNet_2") == int(selection_cfg["decayModePNet_2_eq"]))

    if "idDeepTau2018v2p5VSjet_2_min" in selection_cfg and (
        not available_columns or "idDeepTau2018v2p5VSjet_2" in available_columns
    ):
        _and(ds.field("idDeepTau2018v2p5VSjet_2") >= int(selection_cfg["idDeepTau2018v2p5VSjet_2_min"]))

    if "idDeepTau2018v2p5VSe_2_min" in selection_cfg and (
        not available_columns or "idDeepTau2018v2p5VSe_2" in available_columns
    ):
        _and(ds.field("idDeepTau2018v2p5VSe_2") >= int(selection_cfg["idDeepTau2018v2p5VSe_2_min"]))

    if "idDeepTau2018v2p5VSmu_2_min" in selection_cfg and (
        not available_columns or "idDeepTau2018v2p5VSmu_2" in available_columns
    ):
        _and(ds.field("idDeepTau2018v2p5VSmu_2") >= int(selection_cfg["idDeepTau2018v2p5VSmu_2_min"]))

    if "iso_1_max" in selection_cfg and (not available_columns or "iso_1" in available_columns):
        _and(ds.field("iso_1") < float(selection_cfg["iso_1_max"]))

    if "ip_LengthSig_1_abs_min" in selection_cfg and (
        not available_columns or "ip_LengthSig_1" in available_columns
    ):
        m = float(selection_cfg["ip_LengthSig_1_abs_min"])
        _and((ds.field("ip_LengthSig_1") > m) | (ds.field("ip_LengthSig_1") < -m))

    return expr


def SELECT(df_or_path, plotter_config: dict):
    selection_cfg = (plotter_config or {}).get("selection", {})

    if isinstance(df_or_path, str):
        df = pd.read_parquet(df_or_path)
    else:
        df = df_or_path

    return df[selection_mask(df, selection_cfg)]


def plotting_columns(schema_or_df, plotter_config: dict) -> dict[str, list[str]]:
    plotting_cfg = (plotter_config or {}).get("plotting", {})
    control = plotting_cfg.get("control", [])
    resolution = plotting_cfg.get("resolution", [])

    if hasattr(schema_or_df, "columns"):
        cols = set(schema_or_df.columns)
    else:
        cols = set(schema_or_df)

    columns_control = [c for c in control if c in cols]
    columns_resol = [c for c in resolution if c in cols]

    return {"control": columns_control, "resolution": columns_resol}
