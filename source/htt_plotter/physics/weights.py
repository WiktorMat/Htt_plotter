from __future__ import annotations

import logging


def compute_mc_weight(sample_name: str, params: dict, cache: dict | None = None) -> float:
    if cache is not None and sample_name in cache:
        return cache[sample_name]

    params = params or {}
    sample_params = params.get(sample_name)

    if not sample_params:
        logging.warning("No params for sample: %s -> using weight=1.0", sample_name)
        weight = 1.0
        if cache is not None:
            cache[sample_name] = weight
        return weight

    xs = sample_params.get("xs", 1.0)
    eff = sample_params.get("eff", 1.0)
    filter_eff = sample_params.get("filter_efficiency", 1.0)

    lumi = params.get("lumi", 1.0)

    if "lumi" not in params:
        logging.warning("No lumi in params → using 1.0")

    if eff == 0:
        logging.warning("eff=0 for %s → weight=0", sample_name)
        return 0.0

    weight = (xs * lumi * filter_eff) / eff

    if cache is not None:
        cache[sample_name] = weight

    return weight