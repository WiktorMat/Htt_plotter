from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from htt_plotter.backgrounds.qcd import add_qcd_from_ss
from htt_plotter.config.loader import load_configs
from htt_plotter.io.data_access import DataAccess
from htt_plotter.physics.weights import compute_mc_weight
from htt_plotter.plotting.accumulate import add_histogram
from htt_plotter.plotting.binning import get_binning
from htt_plotter.plotting.colors import get_sample_color
from htt_plotter.plotting.pairs import make_resolution_pairs
from htt_plotter.plotting.render import save_data_mc_ratio_plot, save_stacked_plot
from htt_plotter.selection.selection import make_arrow_filter, selection_columns_used
from htt_plotter.utils.fs import ensure_dir


class Plotter:
    """Plotter orchestrator.

    Performance-oriented design:
    - builds a per-sample index (not per-file)
    - schema is computed once per sample
    - reads parquet via pyarrow.dataset in record batches
    - can fill all histograms in a single pass over the data
    """

    def __init__(
        self,
        xlim_contrl: float | None = None,
        xlim_resolution: float | None = None,
        bins: int = 20,
        alpha: float = 1,
        config_name: str = "config_0",
    ):
        self.xlim_ctrl = xlim_contrl or 100
        self.xlim_resol = xlim_resolution or 50
        self.bins = bins
        self.alpha = alpha

        self.project_root = Path(__file__).resolve().parents[3]

        self.index: list[dict] = []  # per-sample index

        self._bin_cache: dict = {}
        self._mc_weight_cache: dict = {}

        self.contr_name: list[str] = []
        self.recon_name: list[str] = []
        self.resolution_pairs: list[tuple[str, str]] = []

        self.config_name = config_name
        (
            self.sample_config,
            self.params,
            self.variable_config,
            self.plotter_config,
        ) = load_configs(self.project_root, self.config_name)

        self.data_access = DataAccess(
            self.project_root,
            self.sample_config,
            log_every_files=200,
        )

        self.logger = logging.getLogger(__name__)

    def get_sample_color(self, sample: str) -> str:
        return get_sample_color(sample, self.sample_config)

    def load_index(self) -> None:
        self.index = self.data_access.build_index()
        n_files = sum(len(i.get("files", [])) for i in self.index)
        self.logger.info("Indexed samples: %d | total files: %d", len(self.index), n_files)

    def set_parameters(self) -> None:
        # Start from config-defined variable groups, then restrict to what exists.
        desired_control = (self.plotter_config.get("plotting") or {}).get("control", [])
        desired_resolution = (self.plotter_config.get("plotting") or {}).get("resolution", [])

        available_cols = set()
        for item in self.index:
            available_cols |= set(item.get("schema", set()))

        self.contr_name = [v for v in desired_control if v in available_cols]
        self.recon_name = [v for v in desired_resolution if v in available_cols]

        self.logger.info("Control vars: %s", self.contr_name)
        self.logger.info("Resolution vars: %s", self.recon_name)

    def batch(self) -> None:
        self.resolution_pairs = make_resolution_pairs(self.contr_name, self.recon_name)

    def _bin_edges(self, var: str) -> np.ndarray:
        _, _, _, edges = get_binning(
            var,
            self.variable_config,
            xlim_ctrl=self.xlim_ctrl,
            xlim_resol=self.xlim_resol,
            bins=self.bins,
            cache=self._bin_cache,
        )
        return edges

    @staticmethod
    def _to_numpy(batch, name: str) -> np.ndarray:
        # Assumes numeric columns; works well for typical parquet physics tables.
        col = batch.column(batch.schema.get_field_index(name))
        return col.to_numpy(zero_copy_only=False)
    
    def _save_histograms_parquet(self, histograms: dict[str, np.ndarray], edges: np.ndarray, out_path: str, plot_type: str, variable: str, ) -> None:
        rows = []

        bin_left = edges[:-1]
        bin_right = edges[1:]
        bin_center = 0.5 * (bin_left + bin_right)

        for sample, counts in histograms.items():
            for i in range(len(counts)):
                rows.append(
                    {
                        "plot_type": plot_type,
                        "variable": variable,
                        "sample": sample,
                        "bin_left": float(bin_left[i]),
                        "bin_right": float(bin_right[i]),
                        "bin_center": float(bin_center[i]),
                        "count": float(counts[i]),
                    }
                )

        if not rows:
            return

        df = pd.DataFrame(rows)

        parquet_path = Path(out_path).with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)

        self.logger.info("Saved histogram parquet: %s", parquet_path)
    

    def run_all(self, *, do_control: bool = True, do_resolution: bool = True, do_mc_data: bool = True) -> None:
        """Fill and render all plots in a single pass over input data."""

        if not self.index:
            self.load_index()
        if not self.contr_name and not self.recon_name:
            self.set_parameters()
        if do_resolution and not self.resolution_pairs:
            self.batch()

        self.logger.info(
            "Starting run_all: control=%s | resolution=%s | mc_data=%s",
            do_control,
            do_resolution,
            do_mc_data,
        )
        self.logger.info(
            "Plot groups: control_vars=%d | resolution_pairs=%d",
            len(self.contr_name),
            len(self.resolution_pairs),
        )

        if do_control:
            self.logger.info("Ensuring output dir: plots/control_plots")
            ensure_dir("plots/control_plots")
        if do_resolution:
            self.logger.info("Ensuring output dir: plots/resolution_plots")
            ensure_dir("plots/resolution_plots")
        if do_mc_data:
            self.logger.info("Ensuring output dir: plots/mc_data_plots")
            ensure_dir("plots/mc_data_plots")

        # Precompute bin edges
        control_edges = {v: self._bin_edges(v) for v in self.contr_name}
        resolution_edges = {pair: self._bin_edges(f"res_{pair[1]}") for pair in self.resolution_pairs}

        # Histogram containers
        control_hists: dict[str, dict[str, np.ndarray]] = {v: {} for v in self.contr_name}
        resolution_hists: dict[tuple[str, str], dict[str, np.ndarray]] = {pair: {} for pair in self.resolution_pairs}

        # For MC/Data agreement: per var → OS/SS → sample → {counts,sumw2}
        agreement: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
            v: {"OS": {}, "SS": {}} for v in self.contr_name
        }
        sample_kinds: dict[str, str] = {}

        selection_cfg = self.plotter_config.get("selection", {}) or {}
        selection_cols = selection_columns_used(selection_cfg)

        self.logger.info("Beginning scan over %d samples", len(self.index))

        for item in self.index:
            sample = item["sample"]
            kind = item.get("kind", "mc")
            scale = item.get("scale", 1.0)
            schema: set[str] = set(item.get("schema", set()))

            self.logger.info(
                "Sample start: %s | kind=%s | files=%d | schema_cols=%d",
                sample,
                kind,
                len(item.get("files", []) or []),
                len(schema),
            )

            sample_kinds[sample] = kind

            # Determine which columns we actually need for this sample.
            present_control = [v for v in self.contr_name if v in schema]
            present_pairs = [(c, r) for (c, r) in self.resolution_pairs if c in schema and r in schema]

            has_work = (
                (do_control and bool(present_control))
                or (do_resolution and bool(present_pairs))
                or (do_mc_data and bool(present_control) and ("os" in schema))
            )
            if not has_work:
                continue

            needed: set[str] = set()
            if do_control:
                needed |= set(present_control)
            if do_resolution:
                for c, r in present_pairs:
                    needed.add(c)
                    needed.add(r)
            if do_mc_data:
                needed |= set(present_control)
                needed.add("os")

            # Columns needed to evaluate selection (for dataset filter).
            needed |= {c for c in selection_cols if c in schema}

            columns = sorted(needed)

            filter_expr = make_arrow_filter(self.plotter_config, schema)

            # Compute constant MC weight once per sample for agreement plots.
            if kind == "data":
                mc_weight = 1.0
            else:
                params_key = (self.sample_config.get(sample) or {}).get("params_key", sample)
                mc_weight = compute_mc_weight(params_key, self.params, cache=self._mc_weight_cache)

            print("BEFORE:", columns)
            columns = [c for c in columns if c in item["schema"]]
            print("AFTER:", columns)

            for batch in self.data_access.iter_batches(item, columns=columns, filter_expr=filter_expr):
                # Control plots
                if do_control and present_control:
                    for var in present_control:
                        values = self._to_numpy(batch, var)
                        mask = np.isfinite(values)
                        if not np.any(mask):
                            continue

                        edges = control_edges[var]
                        counts, _ = np.histogram(values[mask], bins=edges)
                        counts = counts * scale
                        add_histogram(control_hists[var], sample, counts)

                # Resolution plots
                if do_resolution and present_pairs:
                    for c, r in present_pairs:
                        cv = self._to_numpy(batch, c)
                        rv = self._to_numpy(batch, r)

                        mask = np.isfinite(cv) & np.isfinite(rv) & (cv != 0)
                        if not np.any(mask):
                            continue

                        resolution = (rv[mask] - cv[mask]) / cv[mask]
                        edges = resolution_edges[(c, r)]
                        counts, _ = np.histogram(resolution, bins=edges)
                        counts = counts * scale
                        add_histogram(resolution_hists[(c, r)], sample, counts)

                # MC/Data agreement (OS/SS)
                if do_mc_data and present_control and ("os" in schema) and ("os" in batch.schema.names):
                    os_flag = self._to_numpy(batch, "os")

                    for var in present_control:
                        values = self._to_numpy(batch, var)

                        for region_name, region_mask in {"OS": os_flag == 1, "SS": os_flag == 0}.items():
                            mask = region_mask & np.isfinite(values)
                            if not np.any(mask):
                                continue

                            edges = control_edges[var]
                            counts, _ = np.histogram(values[mask], bins=edges)
                            counts = counts * mc_weight
                            sumw2 = counts * (mc_weight**2)

                            region = agreement[var][region_name]
                            if sample not in region:
                                region[sample] = {
                                    "counts": np.zeros(len(edges) - 1, dtype=float),
                                    "sumw2": np.zeros(len(edges) - 1, dtype=float),
                                }

                            region[sample]["counts"] += counts
                            region[sample]["sumw2"] += sumw2

        # Render control
        if do_control:
            for var, hist in control_hists.items():
                if not hist:
                    continue
                out_path = f"plots/control_plots/{var}.png"
                save_stacked_plot(
                    hist,
                    control_edges[var],
                    title=f"Control: {var}",
                    xlabel=var,
                    out_path=out_path,
                    get_color=self.get_sample_color,
                )

                self._save_histograms_parquet(
                    histograms=hist,
                    edges=control_edges[var],
                    out_path=out_path,
                    plot_type="control",
                    variable=var,
                )

                self.logger.info("Saved control plot: %s", out_path)

        # Render resolution
        if do_resolution:
            for (c, r), hist in resolution_hists.items():
                if not hist:
                    continue
                out_path = f"plots/resolution_plots/Resolution_{r}_from_{c}.png"
                save_stacked_plot(
                    hist,
                    resolution_edges[(c, r)],
                    title=f"Resolution: {r} vs {c}",
                    xlabel=f"res_{r}",
                    out_path=out_path,
                    get_color=self.get_sample_color,
                )
                self._save_histograms_parquet(
                    histograms=hist,
                    edges=resolution_edges[(c, r)],
                    out_path=out_path,
                    plot_type="resolution",
                    variable=f"{r}_from_{c}",
                )
                self.logger.info("Saved resolution plot: %s", out_path)

        # Render agreement
        if do_mc_data:
            for var, regions in agreement.items():
                # add QCD per-var
                histograms = {"OS": regions["OS"], "SS": regions["SS"]}

                add_qcd_from_ss(
                    histograms,
                    {"add_qcd_from_ss": True, "qcd_ff": 1.0},
                    sample_kinds,
                )

                samples = histograms["OS"]

                data_counts = None
                mc_samples: dict[str, np.ndarray] = {}

                for name, hist in samples.items():
                    if sample_kinds.get(name, "mc") == "data":
                        if data_counts is None:
                            data_counts = hist["counts"].copy()
                        else:
                            data_counts += hist["counts"]
                    else:
                        mc_samples[name] = hist["counts"].copy()

                if data_counts is None or not mc_samples:
                    continue

                out_path = f"plots/mc_data_plots/MC_vs_Data_{var}.png"
                save_data_mc_ratio_plot(
                    bin_edges=control_edges[var],
                    data_counts=data_counts,
                    mc_samples=mc_samples,
                    out_path=out_path,
                    xlabel=var,
                    get_color=self.get_sample_color,
                )
                combined = {"data": data_counts, **mc_samples}

                self._save_histograms_parquet(
                    histograms=combined,
                    edges=control_edges[var],
                    out_path=out_path,
                    plot_type="mc_data",
                    variable=var,
                )

                self.logger.info("Saved MC/Data plot: %s", out_path)

    # Backward-compatible convenience methods
    def control_plot(self) -> None:
        self.run_all(do_control=True, do_resolution=False, do_mc_data=False)

    def resolution_plot(self) -> None:
        self.run_all(do_control=False, do_resolution=True, do_mc_data=False)

    def Plot_MC_Data_Agrement(self) -> None:
        self.run_all(do_control=False, do_resolution=False, do_mc_data=True)
