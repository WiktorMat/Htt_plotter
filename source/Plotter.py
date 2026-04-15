import sys
import os
import json
import yaml
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from qcd_from_ss import add_qcd_from_ss
from Selection import *
from data_access import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
            logging.StreamHandler(),
            logging.FileHandler("plotter.log")
        ]
)


class Plotter:
    def __init__(self, xlim_contrl=None, xlim_resolution=None, bins=20, alpha=1, config_name="config_0"):
        self.xlim_ctrl = xlim_contrl or 100
        self.xlim_resol = xlim_resolution or 50
        self.bins = bins
        self.alpha = alpha

        self.project_root = Path(__file__).resolve().parent.parent
        self.base_path = (self.project_root / BASE_PATH).resolve()

        self.files_index = []
        self.log_every_files = 200

        self._bin_cache = {}
        self._mc_weight_cache = {}

        self.contr_name = []
        self.recon_name = []
        self.resolution_pairs = []

        from config_loader import load_configs

        self.config_name = config_name
        self.sample_config, self.params, self.variable_config = load_configs(
            self.project_root,
            self.config_name
        )

        self.data_access = DataAccess(
            self.project_root,
            self.sample_config,
            self.log_every_files
        )

    ### Colors
    def get_sample_color(self, sample):
        cfg = self.sample_config.get(sample, {})
        return cfg.get("color", "tab:blue")


    def ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def load_index(self):
        self.files_index = self.data_access.build_index()
        logging.info(f"Indexed files: {len(self.files_index)}")

    def load_dataframe(self, item, columns):
        return self.data_access.load_parquet(
            item["path"],
            columns=columns,
            selector=SELECT
        )

    def get_scale(self, item):
        return item.get("scale", 1.0) if item.get("kind", "mc") != "data" else 1.0

    def unique_list(self, values):
        return list(dict.fromkeys(values))


    def get_binning(self, var):
        if var in self._bin_cache:
            return self._bin_cache[var]

        cfg = self.variable_config.get(var)

        if cfg is None:
            if str(var).startswith("res_"):
                x_min, x_max, nb = -1, 1, 50
            else:
                x_min, x_max, nb = -self.xlim_ctrl, self.xlim_ctrl, self.bins
        else:
            x_min = cfg["x_min"]
            x_max = cfg["x_max"]
            bin_width = cfg["bin_width"]
            nb = int((x_max - x_min) / bin_width)

        edges = np.linspace(x_min, x_max, nb + 1)
        result = (x_min, x_max, nb, edges)

        self._bin_cache[var] = result
        return result


    def apply_selection(self):
        logging.debug("Selection is lazy: applied only while reading")

    def selected_plots(self):
        return {
            "control": ["pt_tau", "eta_tau"],
            "resolution": ["res_pt_tau"]
        }

    def _has_columns(self, item, needed):
        schema = item.get("schema", set())
        return set(needed).issubset(schema)

    ### Set parameters
    def set_parameters(self):
        self.contr_name = []
        self.recon_name = []

        logging.info("Setting parameters")

        for item in self.files_index:
            try:
                cols = list(item["schema"])
                df = self.load_dataframe(item, cols)
                cfg = plotting(df)

                self.contr_name.extend(cfg.get("control", []))
                self.recon_name.extend(cfg.get("resolution", []))

            except Exception as e:
                logging.warning(f"plotting failed for {item['path']}: {e}")

        self.contr_name = self.unique_list(self.contr_name)
        self.recon_name = self.unique_list(self.recon_name)

        logging.info(f"Control vars: {self.contr_name}")
        logging.info(f"Resolution vars: {self.recon_name}")

    ### Batch processing
    def batch(self, batch_size=250):
        self.resolution_pairs = []

        for control_var in self.contr_name:
            suffix = control_var.split("_", 1)[-1]

            for reco_var in self.recon_name:
                if reco_var.split("_", 1)[-1] == suffix:
                    self.resolution_pairs.append((control_var, reco_var))


    def add_histogram(self, container, sample, counts):
        if sample not in container:
            container[sample] = np.zeros_like(counts, dtype=float)

        container[sample] += counts


    def save_stacked_plot(self, hist_dict, edges, title, xlabel, out_path):
        bottom = np.zeros(len(edges) - 1)

        plt.figure(figsize=(8, 6))

        for sample, counts in hist_dict.items():
            plt.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                bottom=bottom,
                align="edge",
                label=sample,
                color=self.get_sample_color(sample),
                edgecolor="black"
            )
            bottom += counts

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Events")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    ### Control plots
    def control_plot(self):
        logging.info("Starting control plots")

        self.ensure_dir("plots/control_plots")

        vars_all = self.contr_name
        hists = {var: {} for var in vars_all}
        bins_map = {var: self.get_binning(var) for var in vars_all}

        for item in self.files_index:
            available = [v for v in vars_all if v in item.get("schema", set())]

            if not available:
                continue

            df = self.load_dataframe(item, available + ["os"])
            sample = item["sample"]
            scale = self.get_scale(item)

            for var in available:
                values = df[var].dropna().to_numpy()

                if len(values) == 0:
                    continue

                x_min, x_max, nb, edges = bins_map[var]

                counts, _ = np.histogram(
                    values,
                    bins=nb,
                    range=(x_min, x_max)
                )

                counts = counts * scale
                self.add_histogram(hists[var], sample, counts)

        for var in vars_all:
            if not hists[var]:
                continue

            _, _, _, edges = bins_map[var]

            self.save_stacked_plot(
                hists[var],
                edges,
                f"Control: {var}",
                var,
                f"plots/control_plots/{var}.png"
            )

        file_path = f"plots/control_plots"

        logging.info(f"Control plots saved to {file_path}")


    ### Resolution plots
    def resolution_plot(self):
        logging.info("Starting resolution plots")

        self.ensure_dir("plots/resolution_plots")

        if not self.resolution_pairs:
            self.batch()

        pair_hists = {}
        pair_bins = {}

        for c, r in self.resolution_pairs:
            _, _, nb, edges = self.get_binning(f"res_{r}")
            pair_hists[(c, r)] = {}
            pair_bins[(c, r)] = edges

        for item in self.files_index:
            available = [
                (c, r)
                for c, r in self.resolution_pairs
                if c in item.get("schema", set())
                and r in item.get("schema", set())
            ]

            if not available:
                continue

            cols = sorted(set([x for pair in available for x in pair] + ["os"]))
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

                self.add_histogram(pair_hists[(c, r)], sample, counts)

        for (c, r), hist in pair_hists.items():
            if not hist:
                continue

            self.save_stacked_plot(
                hist,
                pair_bins[(c, r)],
                f"Resolution: {r} vs {c}",
                f"res_{r}",
                f"plots/resolution_plots/Resolution_{r}_from_{c}.png"
            )

        file_path = "plots/resolution_plots"

        logging.info(f"Resolution plots saved to {file_path}")


    def compute_mc_weight(self, sample_name):
        if sample_name in self._mc_weight_cache:
            return self._mc_weight_cache[sample_name]

        params = self.params.get(sample_name)

        if params is None:
            logging.warning(f"No params for sample: {sample_name}")
            return 1.0

        xs = params.get("xs", 1.0)
        eff = params.get("eff", 1.0)
        lumi = self.params.get("lumi", 1.0)

        if eff == 0:
            return 0.0

        weight = (xs * lumi) / eff
        self._mc_weight_cache[sample_name] = weight

        return weight

    ### MC Data agreement
    def Plot_MC_Data_Agrement(self):
        logging.info("Starting MC/Data plots")

        self.ensure_dir("plots/mc_data_plots")

        for var in self.contr_name:
            _, _, _, bin_edges = self.get_binning(var)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            histograms = {"OS": {}, "SS": {}}
            sample_kinds = {}

            for item in self.files_index:
                if not self._has_columns(item, [var, "os"]):
                    continue

                df = self.load_dataframe(item, SELECTION_COLUMNS)

                if var not in df.columns or "os" not in df.columns:
                    continue

                sample = item["sample"]
                kind = item.get("kind", "mc")
                sample_kinds[sample] = kind

                values = df[var].to_numpy()
                os_flag = df["os"].to_numpy()

                weight = 1.0 if kind == "data" else self.compute_mc_weight(sample)

                for region_name, region_mask in {
                    "OS": os_flag == 1,
                    "SS": os_flag == 0
                }.items():

                    mask = region_mask & np.isfinite(values)

                    if not np.any(mask):
                        continue

                    counts, _ = np.histogram(values[mask], bins=bin_edges)
                    counts = counts * weight
                    sumw2 = counts * (weight ** 2)

                    if sample not in histograms[region_name]:
                        histograms[region_name][sample] = {
                            "counts": np.zeros(len(bin_edges) - 1),
                            "sumw2": np.zeros(len(bin_edges) - 1)
                        }

                    histograms[region_name][sample]["counts"] += counts
                    histograms[region_name][sample]["sumw2"] += sumw2

            add_qcd_from_ss(
                histograms,
                {"add_qcd_from_ss": True, "qcd_ff": 1.0},
                sample_kinds
            )

            samples = histograms["OS"]

            data_counts = None
            mc_samples = {}

            for name, hist in samples.items():
                if sample_kinds.get(name, "mc") == "data":
                    if data_counts is None:
                        data_counts = hist["counts"].copy()
                    else:
                        data_counts += hist["counts"]
                else:
                    mc_samples[name] = hist["counts"].copy()

            if data_counts is None:
                logging.warning(f"No data for {var}")
                continue

            total_mc = np.zeros_like(data_counts)

            for vals in mc_samples.values():
                total_mc += vals

            if np.all(total_mc == 0):
                continue

            fig = plt.figure(figsize=(7, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

            ax = fig.add_subplot(gs[0])
            rax = fig.add_subplot(gs[1], sharex=ax)

            bottom = np.zeros_like(total_mc)

            for name, vals in mc_samples.items():
                color = "gray" if name == "QCD" else self.get_sample_color(name)
                label = "QCD (from SS)" if name == "QCD" else name

                ax.bar(
                    bin_edges[:-1],
                    vals,
                    width=np.diff(bin_edges),
                    bottom=bottom,
                    align="edge",
                    color=color,
                    label=label
                )

                bottom += vals

            ax.errorbar(
                bin_centers,
                data_counts,
                yerr=np.sqrt(data_counts),
                fmt="o",
                color="black",
                label="Data"
            )

            ax.step(
                bin_edges[:-1],
                total_mc,
                where="post",
                color="black",
                linewidth=1,
                label="MC total"
            )

            ax.set_ylabel("Events")
            ax.legend()

            ratio = np.divide(
                data_counts,
                total_mc,
                out=np.zeros_like(data_counts, dtype=float),
                where=total_mc != 0
            )

            rax.axhline(1.0, linestyle="--", color="black")
            rax.plot(bin_centers, ratio, "o", color="black")
            rax.set_ylabel("Data/MC")
            rax.set_xlabel(var)
            rax.set_ylim(0.5, 1.5)

            out_path = f"plots/mc_data_plots/MC_vs_Data_{var}.png"

            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            logging.info(f"Saved: {out_path}")