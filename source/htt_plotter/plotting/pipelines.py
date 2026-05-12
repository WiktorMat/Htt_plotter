from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from htt_plotter.backgrounds.qcd import add_qcd_from_ss, ensure_qcd_placeholder
from htt_plotter.core.draw_order import order_mapping_by_list, order_mc_samples
from htt_plotter.io.hist_parquet import write_histograms_parquet
from htt_plotter.plotting.render import save_data_mc_ratio_plot, save_stacked_plot


def render_control_plots(
    control_hists: dict[str, dict[str, np.ndarray]],
    control_edges: dict[str, np.ndarray],
    *,
    process_draw_order: list[str],
    get_color: Callable[[str], str],
    layout: str,
    logger: logging.Logger | None = None,
) -> None:
    for var, hist in (control_hists or {}).items():
        if not hist:
            continue
        out_path = f"plots/control_plots/{var}.png"
        save_stacked_plot(
            order_mapping_by_list(hist, process_draw_order),
            control_edges[var],
            title=f"Control: {var}",
            xlabel=var,
            out_path=out_path,
            get_color=get_color,
            layout=layout,
        )

        parquet_path = write_histograms_parquet(
            histograms=hist,
            edges=control_edges[var],
            out_path=out_path,
            plot_type="control",
            variable=var,
        )

        if logger is not None:
            logger.info("Saved control plot: %s (parquet: %s)", out_path, parquet_path)


def render_resolution_plots(
    resolution_hists: dict[tuple[str, str], dict[str, np.ndarray]],
    resolution_edges: dict[tuple[str, str], np.ndarray],
    *,
    process_draw_order: list[str],
    get_color: Callable[[str], str],
    layout: str,
    logger: logging.Logger | None = None,
) -> None:
    for (c, r), hist in (resolution_hists or {}).items():
        if not hist:
            continue
        out_path = f"plots/resolution_plots/Resolution_{r}_from_{c}.png"
        save_stacked_plot(
            order_mapping_by_list(hist, process_draw_order),
            resolution_edges[(c, r)],
            title=f"Resolution: {r} vs {c}",
            xlabel=f"res_{r}",
            out_path=out_path,
            get_color=get_color,
            layout=layout,
        )

        parquet_path = write_histograms_parquet(
            histograms=hist,
            edges=resolution_edges[(c, r)],
            out_path=out_path,
            plot_type="resolution",
            variable=f"{r}_from_{c}",
        )

        if logger is not None:
            logger.info("Saved resolution plot: %s (parquet: %s)", out_path, parquet_path)


def render_mc_data_plots(
    agreement: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    control_edges: dict[str, np.ndarray],
    *,
    process_draw_order: list[str],
    process_kinds: dict[str, str],
    get_color: Callable[[str], str],
    params: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    for var, regions in (agreement or {}).items():
        histograms = {"OS": regions.get("OS", {}), "SS": regions.get("SS", {})}

        def _sum_counts(region_dict: dict[str, dict[str, np.ndarray]], *, want_kind: str) -> float:
            total = 0.0
            for proc_name, h in (region_dict or {}).items():
                if process_kinds.get(proc_name, "mc") != want_kind:
                    continue
                total += float(np.sum(h.get("counts", 0.0)))
            return total

        data_os = _sum_counts(histograms["OS"], want_kind="data")
        data_ss = _sum_counts(histograms["SS"], want_kind="data")
        mc_os = _sum_counts(histograms["OS"], want_kind="mc")
        mc_ss = _sum_counts(histograms["SS"], want_kind="mc")

        lumi = (params or {}).get("lumi") if params is not None else None
        if logger is not None and lumi is not None and data_os > 0:
            data_rate = data_os / lumi
            mc_total = mc_os
            logger.info(
                "[DEBUG NORM] %s | Data/Lumi = %.6e | Sum(MC weights) = %.6e | Ratio (MC / (Data/Lumi)) = %.3f",
                var,
                data_rate,
                mc_total,
                mc_total / data_rate if data_rate > 0 else -1,
            )

        if logger is not None and data_os > 0:
            logger.info(
                "MC/Data totals (%s): data(OS=%.3g,SS=%.3g) mc(OS=%.3g,SS=%.3g) mc/data(OS)=%.3g",
                var,
                data_os,
                data_ss,
                mc_os,
                mc_ss,
                mc_os / data_os,
            )

        add_qcd_from_ss(
            histograms,
            {"add_qcd_from_ss": True, "qcd_ff": 1.0},
            process_kinds,
        )

        samples = histograms["OS"]

        data_counts = None
        data_sumw2 = None
        mc_samples: dict[str, np.ndarray] = {}
        mc_sumw2_total = None

        for name, hist in (samples or {}).items():
            if process_kinds.get(name, "mc") == "data":
                if data_counts is None:
                    data_counts = hist["counts"].copy()
                    data_sumw2 = hist.get("sumw2", hist["counts"]).copy()
                else:
                    data_counts += hist["counts"]
                    data_sumw2 += hist.get("sumw2", hist["counts"])
            else:
                mc_samples[name] = hist["counts"].copy()
                sumw2 = hist.get("sumw2")
                if sumw2 is None:
                    continue
                if mc_sumw2_total is None:
                    mc_sumw2_total = sumw2.copy()
                else:
                    mc_sumw2_total += sumw2

        if data_counts is None:
            continue

        data_unc = None
        if data_sumw2 is not None:
            data_unc = np.sqrt(np.maximum(data_sumw2, 0.0))

        mc_total_unc = None
        if mc_sumw2_total is not None:
            mc_total_unc = np.sqrt(np.maximum(mc_sumw2_total, 0.0))

        mc_samples = ensure_qcd_placeholder(
            mc_samples,
            process_draw_order,
            data_counts,
        )

        out_path = f"plots/mc_data_plots/MC_vs_Data_{var}.png"
        
        # Print overall MC/Data ratio
        total_data = float(np.sum(data_counts))
        total_mc = sum(float(np.sum(h)) for h in mc_samples.values())
        if total_data > 0:
            print(f"[{var}] Total MC/Data ratio: {total_mc / total_data:.3f}")
        
        save_data_mc_ratio_plot(
            bin_edges=control_edges[var],
            data_counts=data_counts,
            mc_samples=order_mc_samples(
                mc_samples,
                desired_order=process_draw_order,
                process_kinds=process_kinds,
            ),
            data_unc=data_unc,
            mc_total_unc=mc_total_unc,
            out_path=out_path,
            xlabel=var,
            get_color=get_color,
        )

        combined = {"data": data_counts, **mc_samples}
        parquet_path = write_histograms_parquet(
            histograms=combined,
            edges=control_edges[var],
            out_path=out_path,
            plot_type="mc_data",
            variable=var,
        )

        if logger is not None:
            logger.info("Saved MC/Data plot: %s (parquet: %s)", out_path, parquet_path)
