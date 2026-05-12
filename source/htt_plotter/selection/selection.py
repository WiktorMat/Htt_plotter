from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

# ============================================================
# 🔧 Operators (single source of truth)
# ============================================================

PANDAS_OPS = {
    ">": lambda x, v: x > v,
    "<": lambda x, v: x < v,
    ">=": lambda x, v: x >= v,
    "<=": lambda x, v: x <= v,
    "==": lambda x, v: x == v,
    "abs<": lambda x, v: x.abs() < v,
    "abs>": lambda x, v: x.abs() > v,
}

ARROW_OPS = {
    ">": lambda f, v: f > v,
    "<": lambda f, v: f < v,
    ">=": lambda f, v: f >= v,
    "<=": lambda f, v: f <= v,
    "==": lambda f, v: f == v,
    "abs<": lambda f, v: (f < v) & (f > -v),
    "abs>": lambda f, v: (f > v) | (f < -v),
}


# ============================================================
# 🧠 Selection config normalization
# ============================================================

def get_selection_cfg(plotter_config: dict | None) -> dict[str, tuple[str, Any]]:
    """
    Expect config like:
    {
        "pt_1": (">", 25),
        "eta_1": ("abs<", 2.1),
    }
    """
    return (plotter_config or {}).get("selection", {})


# ============================================================
# 🟢 Pandas selection
# ============================================================

def selection_mask(df: pd.DataFrame, selection_cfg: dict) -> pd.Series:
    if not selection_cfg:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)

    for col, (op, val) in selection_cfg.items():
        if col not in df.columns:
            continue

        func = PANDAS_OPS.get(op)
        if func is None:
            raise ValueError(f"Unknown operator: {op}")

        mask &= func(df[col], val)

    return mask


def make_selector(plotter_config: dict) -> Callable[[pd.DataFrame], pd.DataFrame]:
    selection_cfg = get_selection_cfg(plotter_config)

    def _selector(df: pd.DataFrame) -> pd.DataFrame:
        return df[selection_mask(df, selection_cfg)]

    return _selector


def SELECT(df_or_path, plotter_config: dict):
    selection_cfg = get_selection_cfg(plotter_config)

    if isinstance(df_or_path, str):
        df = pd.read_parquet(df_or_path)
    else:
        df = df_or_path

    return df[selection_mask(df, selection_cfg)]


# ============================================================
# 🔵 PyArrow filter (pushdown)
# ============================================================

def make_arrow_filter(
    plotter_config: dict,
    available_columns: set[str] | None = None,
):
    selection_cfg = get_selection_cfg(plotter_config)
    if not selection_cfg:
        return None

    try:
        import pyarrow.dataset as ds
    except Exception:
        return None

    expr = None

    for col, (op, val) in selection_cfg.items():
        if available_columns and col not in available_columns:
            continue

        func = ARROW_OPS.get(op)
        if func is None:
            raise ValueError(f"Unknown operator: {op}")

        e = func(ds.field(col), val)
        expr = e if expr is None else (expr & e)

    return expr


# ============================================================
# 📊 Utilities
# ============================================================

def selection_columns_used(selection_cfg: dict) -> list[str]:
    return sorted(selection_cfg.keys())


def plotting_columns(schema_or_df, plotter_config: dict) -> dict[str, list[str]]:
    plotting_cfg = (plotter_config or {}).get("plotting", {})
    control = plotting_cfg.get("control", [])
    resolution = plotting_cfg.get("resolution", [])

    if hasattr(schema_or_df, "columns"):
        cols = set(schema_or_df.columns)
    else:
        cols = set(schema_or_df)

    return {
        "control": [c for c in control if c in cols],
        "resolution": [c for c in resolution if c in cols],
    }