import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from Selection import *
from Config import process, BASE_PATH
import json
import matplotlib.gridspec as gridspec
from qcd_from_ss import add_qcd_from_ss
import yaml


class Plotter:
    def __init__(self, xlim_contrl=None, xlim_resolution=None, bins=20, alpha=1):

        self.xlim_ctrl = xlim_contrl or 100
        self.xlim_resol = xlim_resolution or 50
        self.bins = bins
        self.alpha = alpha
        self.project_root = Path(__file__).resolve().parent.parent
        self.base_path = (self.project_root / BASE_PATH).resolve()
        self.all_data = []
        self.sample_colors = {}

        config_path = self.project_root / "source" / "files.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.sample_config = json.load(f)

        params_path = self.project_root / "source" / "params.yaml"

        with open(params_path, "r", encoding="utf-8") as f:
            self.params = yaml.safe_load(f)

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


    ### Colors
    def get_sample_color(self, sample):
        if sample in self.sample_config:
            if "color" in self.sample_config[sample]:
                return self.sample_config[sample]["color"]

        idx = list(self.sample_config.keys()).index(sample) \
            if sample in self.sample_config else 0

        return self.color_palette[idx % len(self.color_palette)]

    ### Selection
    def apply_selection(self):

        for item in self.all_data:
            df = item["data"]

            try:
                item["filtered_data"] = SELECT(df)
            except Exception as e:
                print(f" Selection failed for {item['name']}: {e}")
                item["filtered_data"] = df

    ### Load data
    def load_data(self):
        self.all_data = []

        for sample_name, cfg in self.sample_config.items():

            kind = cfg.get("kind", "mc")
            scale = cfg.get("scale", 1.0)
            dirs = cfg.get("dirs", [])

            color = self.get_sample_color(sample_name)

            total_files = 0

            for d in dirs:
                path = (self.project_root / d).resolve()

                if not path.exists():
                    print(f" Missing path: {path}")
                    continue

                parquet_files = list(Path(path).rglob("*.parquet"))
                total_files += len(parquet_files)

                print(f"{sample_name}: {len(parquet_files)} files in {path}")

                for file in parquet_files:
                    try:
                        df = pd.read_parquet(file)

                        self.all_data.append({
                            "name": file.name,
                            "data": df,
                            "sample": sample_name,
                            "kind": kind,
                            "scale": scale,
                            "color": color
                        })

                    except Exception as e:
                        print(f"ERROR {file}: {e}")

        print(f"Loaded samples: {set(x['sample'] for x in self.all_data)}")

    ### Set parameters
    def set_parameters(self):
        print("Setting parameters")

        self.contr_name = []
        self.recon_name = []

        for item in self.all_data:
            df = item.get("filtered_data", item["data"])

            try:
                cols = plotting(df)
            except Exception as e:
                print(f" plotting() failed for {item['name']}: {e}")
                continue

            self.contr_name.extend(cols.get("control", []))
            self.recon_name.extend(cols.get("resolution", []))

        self.contr_name = list(dict.fromkeys(self.contr_name))
        self.recon_name = list(dict.fromkeys(self.recon_name))

        print(f"Control vars: {len(self.contr_name)}, {self.contr_name}")
        print(f"Resolution vars: {len(self.recon_name)}, {self.recon_name}")

    ### Batch processing
    def batch(self, batch_size=250):
        print("Creating batches")

        self.batch_dfs = {}

        for contr_name in self.contr_name:
            for recon_name in self.recon_name:

                key = f"{contr_name}_{recon_name}"
                self.batch_dfs[key] = []

                for item in self.all_data:

                    df = item.get("filtered_data", item["data"])

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

    ### Control plots
    def control_plot(self):
        os.makedirs("plots/control_plots", exist_ok=True)

        for contr_name in self.contr_name:

            plt.figure(figsize=(8, 6))

            sample_hists = {}
            bins = None

            for item in self.all_data:

                df = item.get("filtered_data", item["data"])

                if contr_name not in df.columns:
                    continue

                values = df[contr_name].dropna().to_numpy()

                if len(values) == 0:
                    continue

                weights = np.ones(len(values))

                if item.get("kind", "mc") != "data":
                    weights *= item.get("scale", 1.0)

                counts, bins = np.histogram(
                    values,
                    bins=self.bins,
                    range=(-self.xlim_ctrl, self.xlim_ctrl),
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

            bottom = np.zeros(self.bins)

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

        for key, items in self.batch_dfs.items():

            plt.figure()
            used_labels = set()
            bottom = None

            for item in items:

                values = item["data"].values.flatten()
                values = pd.Series(values).dropna()

                if len(values) == 0:
                    continue

                weights = np.ones(len(values))

                if item.get("kind", "mc") != "data":
                    weights *= item.get("scale", 1.0)

                counts, bins = np.histogram(
                    values,
                    bins=self.bins,
                    range=(-self.xlim_resol, self.xlim_resol),
                    weights = weights
                )

                if bottom is None:
                    bottom = np.zeros_like(counts)

                label = item["sample"] if item["sample"] not in used_labels else "_nolegend_"
                used_labels.add(item["sample"])

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom,
                    color=self.get_sample_color(item["sample"]),
                    edgecolor="black",
                    label=label,
                    align="edge"
                )

                bottom += counts

            contr, recon = key.rsplit("_", 1)

            plt.title(f"Resolution: {recon} vs {contr}")
            plt.xlabel(recon)
            plt.ylabel("Events")
            plt.legend()

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

            bin_edges = np.linspace(-self.xlim_ctrl, self.xlim_ctrl, self.bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            histograms = {
                "OS": {},
                "SS": {}
            }

            sample_kinds = {}

            for item in self.all_data:

                df = item.get("filtered_data", item["data"])

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

                    counts, _ = np.histogram(
                        values[mask],
                        bins=bin_edges,
                        weights=weights[mask]
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
    plotter = Plotter(100, 50, 20, 1)
    plotter.load_data()
    plotter.apply_selection()
    plotter.set_parameters()
    plotter.batch(batch_size=250)
    plotter.control_plot()
    plotter.resolution_plot()
    plotter.Plot_MC_Data_Agrement()