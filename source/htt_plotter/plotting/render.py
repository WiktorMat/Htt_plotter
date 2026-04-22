from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_stacked_plot(
    histograms,
    edges,
    title,
    xlabel,
    out_path,
    get_color,
    alpha=1.0,
    layout="stacked",
):
    labels = list(histograms.keys())
    values = [histograms[k] for k in labels]
    colors = [get_color(k) for k in labels]

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)

    fig, ax = plt.subplots(figsize=(8,6))

    if layout == "stacked":
        bottom = np.zeros(len(edges)-1)
        for label, counts, color in zip(labels, values, colors):
            ax.bar(
                centers,
                counts,
                width=width,
                bottom=bottom,
                color=color,
                label=label,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.8,
                align="center",
            )
            bottom += counts

    elif layout == "overlay":
        for label, counts, color in zip(labels, values, colors):
            ax.bar(
                centers,
                counts,
                width=width,
                color=color,
                label=label,
                alpha=min(alpha, 0.5),
                edgecolor="black",
                linewidth=0.8,
                align="center",
            )

    elif layout == "side_by_side":
        labels = list(histograms.keys())
        values = [histograms[k] for k in labels]
        colors = [get_color(k) for k in labels]

        n = len(labels)

        fig = plt.figure(figsize=(5 * n, 5))
        gs = gridspec.GridSpec(1, n, wspace=0.25)

        for i, (label, counts, color) in enumerate(zip(labels, values, colors)):
            ax = fig.add_subplot(gs[0, i])

            centers = 0.5 * (edges[:-1] + edges[1:])
            width = np.diff(edges)

            ax.bar(
                centers,
                counts,
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=alpha,
            )

            ax.set_title(label)
            ax.set_xlabel(xlabel)
            if i == 0:
                ax.set_ylabel("Entries")
            else:
                ax.set_ylabel("")

        fig.suptitle(title)
        plt.tight_layout()

    else:
        raise ValueError(f"Unknown layout: {layout}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_data_mc_ratio_plot(
    *,
    bin_edges: np.ndarray,
    data_counts: np.ndarray,
    mc_samples: dict[str, np.ndarray],
    data_unc: np.ndarray | None = None,
    mc_total_unc: np.ndarray | None = None,
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

    # Ensure QCD is drawn first so it appears at the bottom of the stack.
    items = list(mc_samples.items())
    items.sort(key=lambda kv: (0 if kv[0].lower() == "qcd" else 1))

    for name, vals in items:
        color = "green" if name.lower() == "qcd" else get_color(name)
        label = "QCD (from SS)" if name.lower() == "qcd" else name

        ax.bar(
            bin_edges[:-1],
            vals,
            width=np.diff(bin_edges),
            bottom=bottom,
            align="edge",
            color=color,
            label=label,
            edgecolor="black",
            linewidth=0.8,
        )

        bottom += vals

    # Uncertainties
    if data_unc is None:
        data_unc = np.sqrt(np.maximum(data_counts, 0.0))
    if mc_total_unc is None:
        # Fallback: assume unweighted Poisson statistics.
        mc_total_unc = np.sqrt(np.maximum(total_mc, 0.0))

    # Draw total MC(+QCD) uncertainty band on the main axis (hatched).
    mc_low = np.maximum(total_mc - mc_total_unc, 0.0)
    mc_high = total_mc + mc_total_unc
    x_step = bin_edges
    y1 = np.r_[mc_low, mc_low[-1]]
    y2 = np.r_[mc_high, mc_high[-1]]
    ax.fill_between(
        x_step,
        y1,
        y2,
        step="post",
        facecolor="none",
        edgecolor="black",
        hatch="////",
        linewidth=0.0,
        alpha=0.4,
        label="_nolegend_",
        zorder=2,
    )

    ax.errorbar(
        bin_centers,
        data_counts,
        yerr=data_unc,
        fmt="o",
        color="black",
        label="Data",
        markersize=3.0,
        elinewidth=1.0,
        capsize=1.5,
        capthick=1.0,
        zorder=3,
    )

    ax.set_ylabel("Events")
    ax.legend()

    ratio = np.divide(
        data_counts,
        total_mc,
        out=np.zeros_like(data_counts, dtype=float),
        where=total_mc != 0,
    )

    ratio_unc = np.divide(
        data_unc,
        total_mc,
        out=np.zeros_like(data_unc, dtype=float),
        where=total_mc != 0,
    )

    rel_mc_unc = np.divide(
        mc_total_unc,
        total_mc,
        out=np.zeros_like(mc_total_unc, dtype=float),
        where=total_mc != 0,
    )

    rax.axhline(1.0, linestyle="--", color="black")

    # MC uncertainty band in ratio: 1 +/- (sigma_mc / mc)
    rlow = 1.0 - rel_mc_unc
    rhigh = 1.0 + rel_mc_unc
    rax.fill_between(
        x_step,
        np.r_[rlow, rlow[-1]],
        np.r_[rhigh, rhigh[-1]],
        step="post",
        facecolor="none",
        edgecolor="black",
        hatch="////",
        linewidth=0.0,
        alpha=0.4,
        zorder=1,
    )

    rax.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_unc,
        fmt="o",
        color="black",
        markersize=3.0,
        elinewidth=1.0,
        capsize=1.5,
        capthick=1.0,
        zorder=2,
    )
    rax.set_ylabel("Data/MC")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0.5, 1.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
