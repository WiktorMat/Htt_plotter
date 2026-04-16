from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from htt_plotter.backgrounds.qcd import add_qcd_from_ss
from htt_plotter.config.loader import load_configs
from htt_plotter.io.data_access import DataAccess
from htt_plotter.physics.weights import compute_mc_weight
from htt_plotter.plotting.accumulate import add_histogram
from htt_plotter.plotting.binning import get_binning
from htt_plotter.plotting.colors import get_sample_color
from htt_plotter.plotting.pairs import make_resolution_pairs
from htt_plotter.plotting.render import save_data_mc_ratio_plot, save_stacked_plot
from htt_plotter.selection.selection import SELECTION_COLUMNS, make_selector, plotting_columns
from htt_plotter.utils.fs import ensure_dir


class Plotter:
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

        self.files_index: list[dict] = []
        self.log_every_files = 200

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

        self.selector = make_selector(self.plotter_config)

        self.data_access = DataAccess(
            self.project_root,
            self.sample_config,
            self.log_every_files,
        )

        self.logger = logging.getLogger(__name__)

    def _unique_list(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _present_columns(self, item: dict, columns: list[str]) -> list[str]:
        schema = item.get("schema", set())
        ordered = self._unique_list(list(columns) + list(SELECTION_COLUMNS))
        return [c for c in ordered if c in schema]

    def get_scale(self, item: dict) -> float:
        return item.get("scale", 1.0) if item.get("kind", "mc") != "data" else 1.0

    def get_sample_color(self, sample: str) -> str:
        return get_sample_color(sample, self.sample_config)

    def load_index(self) -> None:
        self.files_index = self.data_access.build_index()
        self.logger.info("Indexed files: %d", len(self.files_index))

    def load_dataframe(self, item: dict, columns: list[str]):
        cols = self._present_columns(item, columns)
        return self.data_access.load_parquet(
            item["path"],
            columns=cols,
            selector=self.selector,
        )

    def set_parameters(self) -> None:
        self.contr_name = []
        self.recon_name = []

        for item in self.files_index:
            cfg = plotting_columns(item.get("schema", set()), self.plotter_config)
            self.contr_name.extend(cfg.get("control", []))
            self.recon_name.extend(cfg.get("resolution", []))

        self.contr_name = self._unique_list(self.contr_name)
        self.recon_name = self._unique_list(self.recon_name)

        self.logger.info("Control vars: %s", self.contr_name)
        self.logger.info("Resolution vars: %s", self.recon_name)

    def batch(self) -> None:
        self.resolution_pairs = make_resolution_pairs(self.contr_name, self.recon_name)

    def control_plot(self) -> None:
        self.logger.info("Starting control plots")

        ensure_dir("plots/control_plots")

        vars_all = self.contr_name
        hists = {var: {} for var in vars_all}
        bins_map = {
            var: get_binning(
                var,
                self.variable_config,
                xlim_ctrl=self.xlim_ctrl,
                xlim_resol=self.xlim_resol,
                bins=self.bins,
                cache=self._bin_cache,
            )
            for var in vars_all
        }

        for item in self.files_index:
            available = [v for v in vars_all if v in item.get("schema", set())]
            if not available:
                continue

            df = self.load_dataframe(item, available)
            sample = item["sample"]
            scale = self.get_scale(item)

            for var in available:
                values = df[var].dropna().to_numpy()
                if len(values) == 0:
                    continue

                x_min, x_max, nb, edges = bins_map[var]
                counts, _ = np.histogram(values, bins=nb, range=(x_min, x_max))
                counts = counts * scale

                add_histogram(hists[var], sample, counts)

        for var in vars_all:
            if not hists[var]:
                continue

            _, _, _, edges = bins_map[var]
            save_stacked_plot(
                hists[var],
                edges,
                title=f"Control: {var}",
                xlabel=var,
                out_path=f"plots/control_plots/{var}.png",
                get_color=self.get_sample_color,
            )

        self.logger.info("Control plots saved to plots/control_plots")

    def resolution_plot(self) -> None:
        self.logger.info("Starting resolution plots")

        ensure_dir("plots/resolution_plots")

        if not self.resolution_pairs:
            self.batch()

        pair_hists: dict[tuple[str, str], dict] = {}
        pair_bins: dict[tuple[str, str], np.ndarray] = {}

        for c, r in self.resolution_pairs:
            _, _, _, edges = get_binning(
                f"res_{r}",
                self.variable_config,
                xlim_ctrl=self.xlim_ctrl,
                xlim_resol=self.xlim_resol,
                bins=self.bins,
                cache=self._bin_cache,
            )
            pair_hists[(c, r)] = {}
            pair_bins[(c, r)] = edges

        for item in self.files_index:
            available = [
                (c, r)
                for c, r in self.resolution_pairs
                if c in item.get("schema", set()) and r in item.get("schema", set())
            ]
            if not available:
                continue

            cols = sorted(set([x for pair in available for x in pair]))
            df = self.load_dataframe(item, cols)

            sample = item["sample"]
            scale = self.get_scale(item)

            for c, r in available:
                cv = df[c].to_numpy()
                rv = df[r].to_numpy()

                mask = np.isfinite(cv) & np.isfinite(rv) & (cv != 0)
                if not np.any(mask):
                    continue

                resolution = (rv[mask] - cv[mask]) / cv[mask]
                edges = pair_bins[(c, r)]

                counts, _ = np.histogram(resolution, bins=edges)
                counts = counts * scale

                add_histogram(pair_hists[(c, r)], sample, counts)

        for (c, r), hist in pair_hists.items():
            if not hist:
                continue

            save_stacked_plot(
                hist,
                pair_bins[(c, r)],
                title=f"Resolution: {r} vs {c}",
                xlabel=f"res_{r}",
                out_path=f"plots/resolution_plots/Resolution_{r}_from_{c}.png",
                get_color=self.get_sample_color,
            )

        self.logger.info("Resolution plots saved to plots/resolution_plots")

    def Plot_MC_Data_Agrement(self) -> None:
        self.logger.info("Starting MC/Data plots")

        ensure_dir("plots/mc_data_plots")

        for var in self.contr_name:
            _, _, _, bin_edges = get_binning(
                var,
                self.variable_config,
                xlim_ctrl=self.xlim_ctrl,
                xlim_resol=self.xlim_resol,
                bins=self.bins,
                cache=self._bin_cache,
            )

            histograms = {"OS": {}, "SS": {}}
            sample_kinds: dict[str, str] = {}

            for item in self.files_index:
                schema = item.get("schema", set())
                if var not in schema or "os" not in schema:
                    continue

                df = self.load_dataframe(item, [var, "os"])
                if var not in df.columns or "os" not in df.columns:
                    continue

                sample = item["sample"]
                kind = item.get("kind", "mc")
                sample_kinds[sample] = kind

                values = df[var].to_numpy()
                os_flag = df["os"].to_numpy()

                if kind == "data":
                    weight = 1.0
                else:
                    params_key = (self.sample_config.get(sample) or {}).get("params_key", sample)
                    weight = compute_mc_weight(
                        params_key,
                        self.params,
                        cache=self._mc_weight_cache,
                    )

                for region_name, region_mask in {"OS": os_flag == 1, "SS": os_flag == 0}.items():
                    mask = region_mask & np.isfinite(values)
                    if not np.any(mask):
                        continue

                    counts, _ = np.histogram(values[mask], bins=bin_edges)
                    counts = counts * weight
                    sumw2 = counts * (weight ** 2)

                    if sample not in histograms[region_name]:
                        histograms[region_name][sample] = {
                            "counts": np.zeros(len(bin_edges) - 1),
                            "sumw2": np.zeros(len(bin_edges) - 1),
                        }

                    histograms[region_name][sample]["counts"] += counts
                    histograms[region_name][sample]["sumw2"] += sumw2

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

            if data_counts is None:
                self.logger.warning("No data for %s", var)
                continue

            out_path = f"plots/mc_data_plots/MC_vs_Data_{var}.png"
            save_data_mc_ratio_plot(
                bin_edges=bin_edges,
                data_counts=data_counts,
                mc_samples=mc_samples,
                out_path=out_path,
                xlabel=var,
                get_color=self.get_sample_color,
            )

            self.logger.info("Saved: %s", out_path)
