from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def calculate_asymmetry(hist_odd: np.ndarray, hist_even: np.ndarray, *, logger: logging.Logger | None = None) -> float:
    diff = hist_odd - hist_even
    sum_abs_diff = float(np.sum(np.abs(diff)))
    total_events = float(np.sum(hist_odd + hist_even))

    asymmetry = sum_abs_diff / total_events if total_events > 0 else 0.0

    if logger is not None:
        logger.info("Sum |odd-even|: %.6f", sum_abs_diff)
        logger.info("Total events: %.6f", total_events)

    return float(asymmetry)


def save_asymmetry_plot(
    *,
    hist_even: np.ndarray,
    hist_odd: np.ndarray,
    cfg: dict[str, Any],
    column: str,
    logger: logging.Logger | None = None,
) -> str:
    asymmetry = calculate_asymmetry(hist_odd, hist_even, logger=logger)

    bins = int(cfg.get("bins", 20))
    range_ = tuple(cfg.get("range", (0, 2 * np.pi)))

    centers = np.linspace(range_[0], range_[1], bins)
    width = (range_[1] - range_[0]) / bins

    plt.figure(figsize=(6, 4))

    plt.bar(centers, hist_even, width=width, alpha=0.7, label="CP-even")
    plt.bar(centers, hist_odd, width=width, alpha=0.7, label="CP-odd")

    plt.xlabel(column)
    plt.ylabel("Events")
    plt.title("Decay Plane Asymmetry")

    plt.text(
        0.05,
        0.95,
        f"Asymmetry = {asymmetry:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    out_dir = str(cfg.get("out_dir", "plots/extra_plots"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    outpath = f"{out_dir}/{column}_asymmetry.png"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    if logger is not None:
        logger.info("Saved asymmetry plot: %s", outpath)

    return outpath
