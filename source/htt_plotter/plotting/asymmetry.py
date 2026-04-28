import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_asymmetry(hist_odd, hist_even, logger=None):
    """Calculates the CP asymmetry value."""
    diff = hist_odd - hist_even
    sum_abs_diff = np.sum(np.abs(diff))
    total_events = np.sum(hist_odd + hist_even)

    asymmetry = sum_abs_diff / total_events if total_events > 0 else 0.0

    if logger:
        logger.info("Sum |odd-even|: %.6f", sum_abs_diff)
        logger.info("Total events: %.6f", total_events)

    return asymmetry

def plot_asymmetry(hist_even, hist_odd, cfg, column, logger=None):
    """Generates and saves the asymmetry plot."""
    asymmetry = calculate_asymmetry(hist_odd, hist_even, logger=logger)

    bins = cfg.get("bins", 20)
    # Zabezpieczenie na wypadek, gdyby range był listą zamiast tupli
    v_range = tuple(cfg.get("range", (0, 2 * np.pi)))

    centers = np.linspace(v_range[0], v_range[1], bins)
    width = (v_range[1] - v_range[0]) / bins

    plt.figure(figsize=(6, 4))

    plt.bar(centers, hist_even, width=width, alpha=0.7, label="CP-even")
    plt.bar(centers, hist_odd, width=width, alpha=0.7, label="CP-odd")

    plt.xlabel(column)
    plt.ylabel("Events")
    plt.title("Decay Plane Asymmetry")
    plt.legend()

    plt.text(
        0.05, 0.95,
        f"Asymmetry = {asymmetry:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(facecolor="white", alpha=0.8)
    )

    out_dir = cfg.get("out_dir", "plots/extra_plots")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    outpath = f"{out_dir}/{column}_asymmetry.png"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    if logger:
        logger.info("Saved asymmetry plot: %s", outpath)