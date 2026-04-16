from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_stacked_plot(
    hist_dict: dict[str, np.ndarray],
    edges: np.ndarray,
    *,
    title: str,
    xlabel: str,
    out_path: str,
    get_color,
    figsize=(8, 6),
) -> None:
    bottom = np.zeros(len(edges) - 1)

    plt.figure(figsize=figsize)

    for sample, counts in hist_dict.items():
        plt.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            bottom=bottom,
            align="edge",
            label=sample,
            color=get_color(sample),
            edgecolor="black",
        )
        bottom += counts

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_data_mc_ratio_plot(
    *,
    bin_edges: np.ndarray,
    data_counts: np.ndarray,
    mc_samples: dict[str, np.ndarray],
    out_path: str,
    xlabel: str,
    get_color,
) -> None:
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total_mc = np.zeros_like(data_counts)
    for vals in mc_samples.values():
        total_mc += vals

    if np.all(total_mc == 0):
        return

    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    bottom = np.zeros_like(total_mc)

    for name, vals in mc_samples.items():
        color = "gray" if name == "QCD" else get_color(name)
        label = "QCD (from SS)" if name == "QCD" else name

        ax.bar(
            bin_edges[:-1],
            vals,
            width=np.diff(bin_edges),
            bottom=bottom,
            align="edge",
            color=color,
            label=label,
        )

        bottom += vals

    ax.errorbar(
        bin_centers,
        data_counts,
        yerr=np.sqrt(data_counts),
        fmt="o",
        color="black",
        label="Data",
    )

    ax.step(
        bin_edges[:-1],
        total_mc,
        where="post",
        color="black",
        linewidth=1,
        label="MC total",
    )

    ax.set_ylabel("Events")
    ax.legend()

    ratio = np.divide(
        data_counts,
        total_mc,
        out=np.zeros_like(data_counts, dtype=float),
        where=total_mc != 0,
    )

    rax.axhline(1.0, linestyle="--", color="black")
    rax.plot(bin_centers, ratio, "o", color="black")
    rax.set_ylabel("Data/MC")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0.5, 1.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
