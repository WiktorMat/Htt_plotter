import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import subprocess
import sys
import json
import matplotlib.gridspec as gridspec

from qcd_from_ss import add_qcd_from_ss
from Selection import *
from data_access import *



class Plotter:
    def __init__(self, xlim_contrl=None, xlim_resolution=None, bins=20, alpha=1):

        self.xlim_ctrl = xlim_contrl or 100
        self.xlim_resol = xlim_resolution or 50
        self.bins = bins
        self.alpha = alpha
        self.project_root = Path(__file__).resolve().parent.parent
        self.base_path = (self.project_root / BASE_PATH).resolve()
        self.sample_colors = {}
        self.files_index = []
        self.files_index = []
        self.log_every_files = 200

        self.project_root = Path(__file__).resolve().parent.parent

        config_dir = self.project_root / "Configurations" / "config_0"

        with open(config_dir / "files.json", "r", encoding="utf-8") as f:
            self.sample_config = json.load(f)

        with open(config_dir / "params.yaml", "r", encoding="utf-8") as f:
            self.params = yaml.safe_load(f)

        with open(config_dir / "variables.json", "r", encoding="utf-8") as f:
            self.variable_config = json.load(f)

        self.data_access = DataAccess(
            self.project_root,
            self.sample_config,
            self.log_every_files
        )

        self.color_palette = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray"
        ]

        self.data_access = DataAccess(self.project_root, self.sample_config)


    ### Colors
    def get_sample_color(self, sample):
        cfg = self.sample_config.get(sample, {})
        return cfg.get("color", "tab:blue")
    
    def get_binning(self, var):

        cfg = self.variable_config.get(var)

        if cfg is None:
            if str(var).startswith("res_"):
                return -1, 1, 50
            return -self.xlim_ctrl, self.xlim_ctrl, self.bins

        x_min = cfg["x_min"]
        x_max = cfg["x_max"]
        bin_width = cfg["bin_width"]

        bins = int((x_max - x_min) / bin_width)

        return x_min, x_max, bins

    ### Selection
    def apply_selection(self):
        print("Selection is lazy: applied only while reading")

    def selected_plots():
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

        print("Setting parameters")

        for item in self.files_index:
            try:
                cols = list(item.get("schema", []))
                df = self.data_access.load_parquet(item["path"], columns=cols, selector=SELECT)
                cfg = plotting(df)
                self.contr_name.extend(cfg.get("control", []))
                self.recon_name.extend(cfg.get("resolution", []))
            except Exception as e:
                print(f"plotting failed for {item['path']}: {e}")

        self.contr_name = list(dict.fromkeys(self.contr_name))
        self.recon_name = list(dict.fromkeys(self.recon_name))

        print("Control vars:", self.contr_name)
        print("Resolution vars:", self.recon_name)

    ### Batch processing
    def batch(self, batch_size=250):
        print("Creating batches")

        self.batch_dfs = {}

        for contr_name in self.contr_name:

            suffix = contr_name.split("_", 1)[-1]

            matching = [
                r for r in self.recon_name
                if r.split("_", 1)[-1] == suffix
            ]

            for recon_name in matching:

                key = f"{contr_name}_{recon_name}"
                self.batch_dfs[key] = []

                for item in self.files_index:

                    df = self.data_access.load_parquet(
                        item["path"],
                        columns=self.get_needed_columns(),
                        selector=SELECT
                    )
                    
                    needed = [contr_name, recon_name, "os"]
                    if not self._has_columns(item, [contr_name, recon_name]):
                        continue

                    df = self.data_access.load_parquet(
                        item["path"], columns=needed, selector=SELECT
                    )

                    if contr_name not in df.columns or recon_name not in df.columns:
                        continue

                    c = df[contr_name]
                    r = df[recon_name]

                    mask = c.notna() & r.notna() & (c != 0)

                    c = c[mask]
                    r = r[mask]

                    if len(c) == 0:
                        continue

                    resolution = (r - c) / c
                    resolution = resolution.replace([np.inf, -np.inf], np.nan).dropna()

                    if len(resolution) == 0:
                        continue

                    batches = [
                        resolution.iloc[i:i + batch_size].reset_index(drop=True)
                        for i in range(0, len(resolution), batch_size)
                    ]

                    self.batch_dfs[key].append({
                        "name": item["name"],
                        "data": pd.concat(batches, axis=1),
                        "sample": item["sample"],
                        "kind": item["kind"],
                        "scale": item["scale"]
                    })

    def get_needed_columns(self):
        return list(set(
            self.contr_name +
            self.recon_name +
            ["os"]
        ))
    
    def load_index(self):
        self.files_index = self.data_access.build_index()
        print(f"Indexed files: {len(self.files_index)}")
    

    ### Control plots
    def control_plot(self):
        os.makedirs("plots/control_plots", exist_ok=True)

        for contr_name in self.contr_name:

            plt.figure(figsize=(8, 6))

            sample_hists = {}
            bins = None

            for item in self.files_index:

                df = self.data_access.load_parquet(
                    item["path"],
                    columns=self.get_needed_columns(),
                    selector=SELECT
                )

                needed = [contr_name, "os"]
                if not self._has_columns(item, [contr_name]):
                    continue

                df = self.data_access.load_parquet(
                    item["path"], columns=needed, selector=SELECT
                )

                if contr_name not in df.columns:
                    continue

                values = df[contr_name].dropna().to_numpy()

                if len(values) == 0:
                    continue

                weights = np.ones(len(values))

                if item.get("kind", "mc") != "data":
                    weights *= item.get("scale", 1.0)

                x_min, x_max, bins_cfg = self.get_binning(contr_name)

                counts, bins = np.histogram(
                    values,
                    bins=bins_cfg,
                    range=(x_min, x_max),
                    weights=weights
                )

                sample = item["sample"]

                if sample not in sample_hists:
                    sample_hists[sample] = counts.astype(float)
                else:
                    sample_hists[sample] += counts.astype(float)

            if not sample_hists:
                print(f"No data for {contr_name}")
                plt.close()
                continue

            bottom = np.zeros(len(bins) - 1)

            for sample, counts in sample_hists.items():

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom,
                    color=self.get_sample_color(sample),
                    edgecolor="black",
                    label=sample,
                    align="edge"
                )

                bottom += counts

            plt.title(f"Control: {contr_name}")
            plt.xlabel(contr_name)
            plt.ylabel("Events")
            plt.legend()
            plt.grid(True, alpha=0.3)

            out_path = f"plots/control_plots/{contr_name}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Control plots saved to: {out_path}")

    ### Resolution plots
    def resolution_plot(self):
        os.makedirs("plots/resolution_plots", exist_ok=True)

        for key in self.batch_dfs.keys():

            contr, recon = key.rsplit("_", 1)
            res_var = f"res_{recon}"

            x_min, x_max, bins_cfg = self.get_binning(res_var)
            bin_edges = np.linspace(x_min, x_max, bins_cfg + 1)

            sample_hists = {}

            for item in self.batch_dfs[key]:

                values = item["data"].values.flatten()
                values = pd.Series(values).dropna()

                if len(values) == 0:
                    continue

                weights = np.ones(len(values))

                if item.get("kind", "mc") != "data":
                    weights *= item.get("scale", 1.0)

                counts, _ = np.histogram(
                    values,
                    bins=bin_edges,
                    range=(x_min, x_max),
                    weights=weights
                )

                sample = item["sample"]

                if sample not in sample_hists:
                    sample_hists[sample] = counts.astype(float)
                else:
                    sample_hists[sample] += counts.astype(float)

            if not sample_hists:
                print(f"No data for {key}")
                continue

            plt.figure()
            bottom = np.zeros(len(bin_edges) - 1)

            for sample, counts in sample_hists.items():

                plt.bar(
                    bin_edges[:-1],
                    counts,
                    width=np.diff(bin_edges),
                    bottom=bottom,
                    color=self.get_sample_color(sample),
                    edgecolor="black",
                    label=sample,
                    align="edge"
                )

                bottom += counts

            plt.title(f"Resolution: {recon} vs {contr}")
            plt.xlabel(res_var)
            plt.ylabel("Events")
            plt.legend()
            plt.grid(True, alpha=0.3)

            out_path = f"plots/resolution_plots/Resolution_{recon}_from_{contr}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()

            print(f"Resolution plots saved to: {out_path}")

    def compute_mc_weight(self, sample_name):
        p = self.params.get(sample_name, None)

        if p is None:
            print(f"WARN No params for sample: {sample_name}")
            return 1.0

        xs = p.get("xs", 1.0)
        eff = p.get("eff", 1.0)
        lumi = self.params.get("lumi", 1.0)

        if eff == 0:
            return 0.0

        return (xs * lumi) / eff

    def Plot_MC_Data_Agrement(self):
        os.makedirs("plots/mc_data_plots", exist_ok=True)
        used_labels = set()

        for var in self.contr_name:

            x_min, x_max, bins_cfg = self.get_binning(var)

            bin_edges = np.linspace(-self.xlim_ctrl, self.xlim_ctrl, self.bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            histograms = {
                "OS": {},
                "SS": {}
            }

            sample_kinds = {}

            total_files = len(self.files_index)
            processed_files = 0

            for item in self.files_index:

                needed = [var, "os"]  # CHANGED
                if not self._has_columns(item, needed):
                    continue

                df = self.data_access.load_parquet(
                    item["path"], columns=needed, selector=SELECT
                )

                processed_files += 1

                if processed_files % 200 == 0 or processed_files == total_files:
                    file_name = Path(item["path"]).name
                    folder_name = Path(item["path"]).parent

                    print(
                        f"INFO Processed {processed_files}/{total_files} files\n"
                        f"Current folder: {folder_name}\n"
                        f"Current file: {file_name}"
                    )


                df = self.data_access.load_parquet(
                    item["path"],
                    columns=self.get_needed_columns(),
                    selector=SELECT
                )

                if var not in df.columns or "os" not in df.columns:
                    continue

                sample = item["sample"]
                kind = item.get("kind", "mc")
                scale = item.get("scale", 1.0)

                sample_kinds[sample] = kind

                values = df[var].to_numpy()
                os_flag = df["os"].to_numpy()

                weights = np.ones(len(values))
                if item.get("kind", "mc") != "data":
                    scale = self.compute_mc_weight(item["sample"])
                    weights *= scale

                regions = {
                    "OS": (os_flag == 1),
                    "SS": (os_flag == 0)
                }

                for region_name, mask in regions.items():

                    if not np.any(mask):
                        continue


                    bin_edges = np.linspace(x_min, x_max, bins_cfg + 1)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                    counts, bins = np.histogram(
                        values,
                        bins=bins_cfg,
                        range=(x_min, x_max),
                        weights=weights
                    )

                    sumw2, _ = np.histogram(
                        values[mask],
                        bins=bin_edges,
                        weights=weights[mask] ** 2
                    )

                    if sample not in histograms[region_name]:
                        histograms[region_name][sample] = {
                            "counts": counts,
                            "sumw2": sumw2
                        }
                    else:
                        histograms[region_name][sample]["counts"] += counts
                        histograms[region_name][sample]["sumw2"] += sumw2

            config = {
                "add_qcd_from_ss": True,
                "qcd_ff": 1.0
            }

            add_qcd_from_ss(histograms, config, sample_kinds)

            samples = histograms["OS"]

            data_counts = None
            data_sumw2 = None

            mc_samples = {}

            for name, hist in samples.items():

                counts = hist["counts"]

                if sample_kinds.get(name, "mc") == "data":
                    if data_counts is None:
                        data_counts = counts.copy()
                        data_sumw2 = hist["sumw2"].copy()
                    else:
                        data_counts += counts
                        data_sumw2 += hist["sumw2"]

                else:
                    mc_samples[name] = counts.copy()

            if data_counts is None:
                print("No data found")
                return

            total_mc = np.zeros_like(data_counts)

            for vals in mc_samples.values():
                total_mc += vals


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
                    label=label,
                    color=color
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

            print(f"MC to Data plots saved to: {out_path}")

### MAIN
if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "json_generator.py"],
        cwd=project_root / "source",
        check=True
    )
    plotter = Plotter(100, 50, 20, 1)
    plotter.load_index()
    plotter.apply_selection()
    plotter.set_parameters()
    plotter.batch(batch_size=250)
    plotter.control_plot()
    plotter.resolution_plot()
    plotter.Plot_MC_Data_Agrement()