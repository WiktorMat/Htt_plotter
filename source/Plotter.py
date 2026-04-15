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
        self.log_every_files = 200
        self._bin_cache = {}
        self._mc_weight_cache = {}

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
        if var in self._bin_cache:
            return self._bin_cache[var]

        cfg = self.variable_config.get(var)

        if cfg is None:
            if str(var).startswith("res_"):
                res = (-1, 1, 50)
            else:
                res = (-self.xlim_ctrl, self.xlim_ctrl, self.bins)
        else:
            x_min = cfg["x_min"]
            x_max = cfg["x_max"]
            bin_width = cfg["bin_width"]
            bins = int((x_max - x_min) / bin_width)
            edges = np.linspace(x_min, x_max, bins + 1)
            res = (x_min, x_max, bins, edges)

        self._bin_cache[var] = res
        return res

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
                cols = list(item["schema"])
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
        self.resolution_pairs = []
        for contr_name in self.contr_name:
            suffix = contr_name.split("_",1)[-1]
            for recon_name in self.recon_name:
                if recon_name.split("_",1)[-1] == suffix:
                    self.resolution_pairs.append((contr_name, recon_name))
    

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
        vars_all = self.contr_name
        hists = {v:{} for v in vars_all}
        bins_map = {v:self.get_binning(v) for v in vars_all}

        for item in self.files_index:
            available = [v for v in vars_all if v in item.get("schema", set())]
            if not available:
                continue
            df = self.data_access.load_parquet(item["path"], columns=available+["os"], selector=SELECT)
            sample = item["sample"]
            scale = item.get("scale",1.0) if item.get("kind","mc") != "data" else 1.0

            for var in available:
                vals = df[var].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                x_min, x_max, nb, edges = bins_map[var]
                counts,_ = np.histogram(vals, bins=nb, range=(x_min,x_max))
                counts = counts * scale
                hists[var].setdefault(sample, np.zeros(nb))
                hists[var][sample] += counts

        for var in vars_all:
            if not hists[var]:
                continue
            x_min, x_max, nb, edges = bins_map[var]

            if var in self._bin_cache:
                x_min, x_max, nb, edges = self._bin_cache[var]
            else:
                edges = np.linspace(x_min, x_max, nb + 1)
                self._bin_cache[var] = (x_min, x_max, nb, edges)
            bottom = np.zeros(nb)
            plt.figure(figsize=(8,6))
            for sample,counts in hists[var].items():
                plt.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align='edge', label=sample, color=self.get_sample_color(sample), edgecolor='black')
                bottom += counts
            plt.title(f"Control: {var}")
            plt.xlabel(var)
            plt.ylabel("Events")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(f"plots/control_plots/{var}.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Control plots saved to: control plots")

    ### Resolution plots
    def resolution_plot(self):
        os.makedirs("plots/resolution_plots", exist_ok=True)

        if not hasattr(self, "resolution_pairs"):
            self.batch()


        pair_hists = {}
        pair_bins = {}

        for c,r in self.resolution_pairs:
            res_var = f"res_{r}"
            x_min,x_max,nb = self.get_binning(res_var)
            pair_hists[(c,r)] = {}
            key = (c, r)

            if key in self._bin_cache:
                edges = self._bin_cache[key][3]
            else:
                edges = np.linspace(x_min, x_max, nb + 1)
                self._bin_cache[key] = (x_min, x_max, nb, edges)

            pair_bins[key] = edges

        for item in self.files_index:
            available = [(c,r) for c,r in self.resolution_pairs if c in item.get("schema",set()) and r in item.get("schema",set())]
            if not available:
                continue
            cols = sorted(set([x for p in available for x in p] + ["os"]))
            df = self.data_access.load_parquet(item["path"], columns=cols, selector=SELECT)
            sample = item["sample"]
            scale = item.get("scale",1.0) if item.get("kind","mc") != "data" else 1.0

            for c,r in available:
                cv = df[c].to_numpy(); rv = df[r].to_numpy()
                mask = np.isfinite(cv) & np.isfinite(rv) & (cv != 0)
                if not np.any(mask):
                    continue
                res = (rv[mask]-cv[mask])/cv[mask]
                edges = pair_bins[(c,r)]
                counts,_ = np.histogram(res, bins=edges)
                counts *= scale
                pair_hists[(c,r)].setdefault(sample, np.zeros(len(edges)-1))
                pair_hists[(c,r)][sample] += counts

        for (c,r),samples in pair_hists.items():
            if not samples:
                continue
            edges = pair_bins[(c,r)]
            bottom = np.zeros(len(edges)-1)
            plt.figure(figsize=(8,6))
            for sample,counts in samples.items():
                plt.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align='edge', label=sample, color=self.get_sample_color(sample), edgecolor='black')
                bottom += counts
            plt.title(f"Resolution: {r} vs {c}")
            plt.xlabel(f"res_{r}")
            plt.ylabel("Events")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(f"plots/resolution_plots/Resolution_{r}_from_{c}.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Resolution plots saved to: resolution_plots")

    def compute_mc_weight(self, sample_name):

        if sample_name in self._mc_weight_cache:
            return self._mc_weight_cache[sample_name]

        p = self.params.get(sample_name, None)

        if p is None:
            print(f"WARN No params for sample: {sample_name}")
            return 1.0

        xs = p.get("xs", 1.0)
        eff = p.get("eff", 1.0)
        lumi = self.params.get("lumi", 1.0)

        if eff == 0:
            return 0.0

        w = (xs * lumi) / eff
        self._mc_weight_cache[sample_name] = w

        return w

    def Plot_MC_Data_Agrement(self):
        os.makedirs("plots/mc_data_plots", exist_ok=True)

        for var in self.contr_name:

            x_min, x_max, bins_cfg, bin_edges = self.get_binning(var)
            key = ("mc_data", var)

            if key in self._bin_cache:
                bin_edges = self._bin_cache[key][3]
            else:
                self._bin_cache[key] = (x_min, x_max, bins_cfg, bin_edges)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            histograms = {
                "OS": {},
                "SS": {}
            }

            sample_kinds = {}

            total_files = len(self.files_index)
            processed_files = 0

            for item in self.files_index:

                if var not in item.get("schema", set()) or "os" not in item.get("schema", set()):
                    continue

                df = self.data_access.load_parquet(
                    item["path"],
                    columns=SELECTION_COLUMNS,
                    selector=SELECT
                )

                if var not in df.columns or "os" not in df.columns:
                    continue

                processed_files += 1

                if processed_files % 200 == 0 or processed_files == total_files:
                    print(
                        f"INFO Processed {processed_files}/{total_files} files\n"
                        f"File: {Path(item['path']).name}"
                    )

                sample = item["sample"]
                kind = item.get("kind", "mc")
                sample_kinds[sample] = kind

                values = df[var].to_numpy()
                os_flag = df["os"].to_numpy()

                if kind == "data":
                    weight = 1.0
                else:
                    weight = self.compute_mc_weight(sample)

                regions = {
                    "OS": os_flag == 1,
                    "SS": os_flag == 0
                }

                for region_name, region_mask in regions.items():

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

                if sample_kinds.get(name, "mc") == "data":
                    if data_counts is None:
                        data_counts = hist["counts"].copy()
                        data_sumw2 = hist["sumw2"].copy()
                    else:
                        data_counts += hist["counts"]
                        data_sumw2 += hist["sumw2"]
                else:
                    mc_samples[name] = hist["counts"].copy()

            if data_counts is None:
                print("No data found")
                continue

            total_mc = np.zeros_like(data_counts)

            for vals in mc_samples.values():
                total_mc += vals

            if np.all(total_mc == 0):
                print(f"Skip {var}: empty MC")
                continue

            # =========================
            # PLOT
            # =========================
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
    plotter.control_plot()
    plotter.batch()
    plotter.resolution_plot()
    plotter.Plot_MC_Data_Agrement()