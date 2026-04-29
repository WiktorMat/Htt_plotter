from __future__ import annotations

import numpy as np


def get_binning(
    var: str,
    variable_config: dict,
    *,
    xlim_ctrl: float,
    xlim_resol: float,
    bins: int,
    cache: dict | None = None,
):
    if cache is not None and var in cache:
        return cache[var]

    cfg = (variable_config or {}).get(var)

    if cfg is None:
        if str(var).startswith("res_"):
            if isinstance(xlim_resol, (int, float)):
                x_min, x_max = -xlim_resol, xlim_resol
            else:
                x_min, x_max = xlim_resol
            if bins is not None:
                nb = bins
            else:
                bin_width = 0.05
                nb = int((x_max - x_min) / bin_width)
        else:
            x_min, x_max, nb = -xlim_ctrl, xlim_ctrl, bins
    else:
        x_min = cfg["x_min"]
        x_max = cfg["x_max"]
        bin_width = cfg["bin_width"]
        nb = int((x_max - x_min) / bin_width)

    edges = np.linspace(x_min, x_max, nb + 1)
    result = (x_min, x_max, nb, edges)

    if cache is not None:
        cache[var] = result

    return result
