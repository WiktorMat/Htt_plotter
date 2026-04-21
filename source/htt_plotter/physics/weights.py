from __future__ import annotations

import logging


def compute_mc_weight(sample_name: str, params: dict, cache: dict | None = None) -> float:
    if cache is not None and sample_name in cache:
        return cache[sample_name]

    sample_params = (params or {}).get(sample_name)

    if sample_params is None:
        logging.warning("No params for sample: %s", sample_name)
        weight = 1.0
        if cache is not None:
            cache[sample_name] = weight
        return weight

    xs = sample_params.get("xs", 1.0)
    eff = sample_params.get("eff", 1.0)
    filter_eff = sample_params.get("filter_efficiency", 1.0)
    lumi = (params or {}).get("lumi", 1.0)

    if eff == 0:
        return 0.0

    weight = (xs * lumi * filter_eff) / eff

    if cache is not None:
        cache[sample_name] = weight

    return weight
