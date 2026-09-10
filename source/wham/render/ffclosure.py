"""MUFFIN fake-factor closure plots in the configured determination region."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from wham.config import AnalysisConfig
from wham.qcd import region_sums
from wham.render.common import cms_label, draw_unc_band, save, var_label

PASS, FAIL, NAN_WEIGHT = "pass", "fail", "nan_weight"


def _data_minus_mc(
    h: Any, cfg: AnalysisConfig, region: str
) -> tuple[np.ndarray, np.ndarray]:
    data_proc = cfg.data_process()
    qcd_proc = cfg.qcd_process()
    if data_proc is None:
        nbins = h.axes[-1].size
        return np.zeros(nbins), np.zeros(nbins)
    data, data_w2, mc, mc_w2 = region_sums(
        h,
        region,
        processes=list(cfg.processes.keys()),
        data_proc=data_proc,
        qcd_proc=qcd_proc,
    )
    return data - mc, data_w2 + mc_w2


def closure_metrics(
    target: np.ndarray,
    target_var: np.ndarray,
    prediction: np.ndarray,
    prediction_var: np.ndarray,
    edges: np.ndarray | None = None,
) -> dict[str, object]:
    """Return compact MUFFIN closure metrics.

    The shape chi-square rescales prediction to the target integral and treats
    that scale as fixed, using only diagonal statistical variances. It is meant
    as a plotting diagnostic, not as a formal test with a full covariance matrix.
    """
    target = np.asarray(target, dtype=float)
    target_var = np.asarray(target_var, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    prediction_var = np.asarray(prediction_var, dtype=float)

    target_sum = float(np.sum(target))
    prediction_sum = float(np.sum(prediction))
    eps = 1e-12
    warnings: list[str] = []

    if target_sum > eps:
        norm_delta = (prediction_sum - target_sum) / target_sum
    else:
        norm_delta = None
        warnings.append("target integral is not positive")

    valid_z = np.isfinite(prediction - target) & np.isfinite(prediction_var + target_var)
    valid_z &= (prediction_var + target_var) > 0
    z = np.full_like(target, np.nan, dtype=float)
    z[valid_z] = (prediction[valid_z] - target[valid_z]) / np.sqrt(
        prediction_var[valid_z] + target_var[valid_z]
    )
    if np.any(valid_z):
        max_idx = int(np.nanargmax(np.abs(z)))
        max_abs_z = float(abs(z[max_idx]))
        max_z = float(z[max_idx])
        max_bin = (
            (float(edges[max_idx]), float(edges[max_idx + 1]))
            if edges is not None
            else None
        )
    else:
        max_idx = None
        max_abs_z = None
        max_z = None
        max_bin = None

    if target_sum > eps and abs(prediction_sum) > eps:
        alpha = target_sum / prediction_sum
        shape_var = alpha**2 * prediction_var + target_var
        valid_shape = np.isfinite(alpha * prediction - target) & np.isfinite(shape_var)
        valid_shape &= shape_var > 0
        n_valid_shape = int(np.sum(valid_shape))
        ndf = n_valid_shape - 1
        if ndf > 0:
            chi2 = float(
                np.sum(((alpha * prediction[valid_shape] - target[valid_shape]) ** 2)
                       / shape_var[valid_shape])
            )
            chi2_ndf = chi2 / ndf
        else:
            chi2 = None
            chi2_ndf = None
            warnings.append("not enough valid bins for shape chi-square")
    else:
        alpha = None
        chi2 = None
        chi2_ndf = None
        ndf = 0
        n_valid_shape = 0
        if abs(prediction_sum) <= eps:
            warnings.append("prediction integral is numerically zero")

    return {
        "target_sum": target_sum,
        "prediction_sum": prediction_sum,
        "norm_delta": norm_delta,
        "shape_alpha": alpha,
        "shape_chi2": chi2,
        "shape_ndf": ndf,
        "shape_chi2_ndf": chi2_ndf,
        "shape_valid_bins": n_valid_shape,
        "max_abs_z": max_abs_z,
        "max_z": max_z,
        "max_z_bin": max_bin,
        "max_z_bin_index": max_idx,
        "skipped_shape_bins": int(len(target) - n_valid_shape),
        "target_negative_bins": int(np.sum(target < 0)),
        "target_negative_sum": float(np.sum(target[target < 0])),
        "prediction_negative_bins": int(np.sum(prediction < 0)),
        "prediction_negative_sum": float(np.sum(prediction[prediction < 0])),
        "warnings": warnings,
    }


def metric_lines(metrics: dict[str, object], selected: list[str]) -> list[str]:
    lines = []
    if "normalization" in selected:
        norm = metrics["norm_delta"]
        lines.append(
            "norm: N/A" if norm is None else f"norm: {100.0 * float(norm):+.1f}%"
        )
    if "shape_chi2" in selected:
        chi2_ndf = metrics["shape_chi2_ndf"]
        ndf = metrics["shape_ndf"]
        lines.append(
            "shape chi2/ndf: N/A"
            if chi2_ndf is None
            else f"shape chi2/ndf: {float(chi2_ndf):.2f} ({int(ndf)} dof)"
        )
    if "max_significance" in selected:
        max_abs_z = metrics["max_abs_z"]
        max_bin = metrics["max_z_bin"]
        if max_abs_z is None:
            lines.append("max |z|: N/A")
        elif max_bin is None:
            lines.append(f"max |z|: {float(max_abs_z):.2f}")
        else:
            lo, hi = max_bin
            lines.append(f"max |z|: {float(max_abs_z):.2f} [{lo:g}, {hi:g})")
    return lines


def render_ffclosure(
    cfg: AnalysisConfig, hists: dict, outdir: Path, only_vars=None, console=None
) -> None:
    closure = cfg.fake_factors.closure if cfg.fake_factors is not None else None
    if closure is None or not closure.enabled:
        return

    for (family, var), h in sorted(hists.items()):
        if family != "ffclosure" or (only_vars and var not in only_vars):
            continue
        if not {PASS, FAIL, NAN_WEIGHT} <= set(h.axes["region"]):
            continue

        edges = h.axes[-1].edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        target, target_var = _data_minus_mc(h, cfg, PASS)
        prediction, prediction_var = _data_minus_mc(h, cfg, FAIL)
        target_unc = np.sqrt(np.maximum(target_var, 0.0))
        prediction_unc = np.sqrt(np.maximum(prediction_var, 0.0))
        metrics = closure_metrics(target, target_var, prediction, prediction_var, edges)

        nan_events = 0.0
        for proc in h.axes["process"]:
            view = h[{"process": proc, "region": NAN_WEIGHT, "variation": "nominal"}].view()
            nan_events += float(np.sum(view["value"]))

        if console is not None:
            norm = metrics["norm_delta"]
            chi2 = metrics["shape_chi2_ndf"]
            max_abs_z = metrics["max_abs_z"]
            console.print(
                f"  ff_closure/{var}: target={metrics['target_sum']:.6g}, "
                f"prediction={metrics['prediction_sum']:.6g}, "
                f"norm={'N/A' if norm is None else f'{100.0 * float(norm):+.2f}%'}, "
                f"shape_chi2/ndf={'N/A' if chi2 is None else f'{float(chi2):.3g}'}, "
                f"max|z|={'N/A' if max_abs_z is None else f'{float(max_abs_z):.3g}'}, "
                f"skipped_bins={metrics['skipped_shape_bins']}, "
                f"target_neg={metrics['target_negative_bins']} "
                f"({metrics['target_negative_sum']:.6g}), "
                f"prediction_neg={metrics['prediction_negative_bins']} "
                f"({metrics['prediction_negative_sum']:.6g}), "
                f"nan_ff_events={nan_events:.0f}"
            )
            for warning in metrics["warnings"]:
                console.print(f"  [yellow]ff_closure/{var}: {warning}[/yellow]")

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(gs[0])
        rax = fig.add_subplot(gs[1], sharex=ax)

        ax.stairs(target, edges, color="black", linewidth=2, label="Target: data $-$ MC")
        draw_unc_band(ax, edges, target - target_unc, target + target_unc)
        ax.stairs(
            prediction,
            edges,
            color="tab:olive",
            linewidth=2,
            label="Prediction: MUFFIN-weighted data $-$ MC",
        )
        draw_unc_band(ax, edges, prediction - prediction_unc, prediction + prediction_unc)

        ax.axhline(0.0, color="gray", linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("Events")
        ymax = max(
            float(np.nanmax(target + target_unc)) if len(target) else 0.0,
            float(np.nanmax(prediction + prediction_unc)) if len(prediction) else 0.0,
            0.0,
        )
        ymin = min(
            float(np.nanmin(target - target_unc)) if len(target) else 0.0,
            float(np.nanmin(prediction - prediction_unc)) if len(prediction) else 0.0,
            0.0,
        )
        span = ymax - ymin
        ax.set_ylim(ymin - 0.15 * span, ymax + 0.25 * span if span else 1.0)
        ax.legend(fontsize=16)
        ax.tick_params(labelbottom=False)
        cms_label(ax, cfg)

        lines = metric_lines(metrics, closure.metrics)
        if lines:
            ax.text(
                0.03,
                0.95,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
            )

        safe_target = np.where(target != 0, target, np.nan)
        ratio = prediction / safe_target
        ratio_unc = np.sqrt(
            prediction_var / safe_target**2
            + (prediction**2 * target_var) / safe_target**4
        )
        rax.axhline(1.0, linestyle="--", color="black", linewidth=1)
        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt="o",
                     color="tab:olive", markersize=5)
        rax.set_ylim(0.0, 2.0)
        rax.set_ylabel("Prediction / Target")
        rax.set_xlabel(var_label(cfg, var))

        save(fig, outdir / "ff_closure", var, console)
