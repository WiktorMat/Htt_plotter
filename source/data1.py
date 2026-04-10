import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from selection import *
from config import process, BASE_PATH
import json


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
                print(f"[WARN] Selection failed for {item['name']}: {e}")
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
                    print(f"[WARN] Missing path: {path}")
                    continue

                parquet_files = list(Path(path).rglob("*.parquet"))
                total_files += len(parquet_files)

                print(f"[LOAD] {sample_name}: {len(parquet_files)} files in {path}")

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
                        print(f"[ERROR] {file}: {e}")

        print(f"[INFO] Loaded samples: {set(x['sample'] for x in self.all_data)}")

    ### Set parameters
    def set_parameters(self):
        print("[INFO] Setting parameters...")

        self.contr_name = []
        self.recon_name = []

        for item in self.all_data:
            df = item.get("filtered_data", item["data"])

            try:
                cols = plotting(df)
            except Exception as e:
                print(f"[WARN] plotting() failed for {item['name']}: {e}")
                continue

            self.contr_name.extend(cols.get("control", []))
            self.recon_name.extend(cols.get("resolution", []))

        self.contr_name = list(dict.fromkeys(self.contr_name))
        self.recon_name = list(dict.fromkeys(self.recon_name))

        print(f"[INFO] Control vars: {len(self.contr_name)}")
        print(f"[INFO] Resolution vars: {len(self.recon_name)}")

    ### Batch processing
    def batch(self, batch_size=250):
        print("[INFO] Creating batches...")

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
                        "sample": item["sample"]
                    })

    ### Control plots
    def control_plot(self):
        os.makedirs("control_plots", exist_ok=True)

        for contr_name in self.contr_name:

            plt.figure()
            bottom = None

            for item in self.all_data:

                df = item.get("filtered_data", item["data"])

                if contr_name not in df.columns:
                    continue

                values = df[contr_name].dropna()

                counts, bins = np.histogram(
                    values,
                    bins=self.bins,
                    range=(-self.xlim_ctrl, self.xlim_ctrl)
                )

                if bottom is None:
                    bottom = np.zeros_like(counts)

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom,
                    color=item["color"],
                    edgecolor="black",
                    label=item["sample"],
                    align="edge"
                )

                bottom += counts

            plt.title(f"Control: {contr_name}")
            plt.xlabel(contr_name)
            plt.ylabel("Events")
            plt.legend()
            plt.grid(True)

            plt.savefig(f"control_plots/{contr_name}.png", dpi=300)
            plt.close()

    ### Resolution plots
    def resolution_plot(self):
        os.makedirs("resolution_plots", exist_ok=True)

        for key, items in self.batch_dfs.items():

            plt.figure()
            bottom = None

            for item in items:

                values = item["data"].values.flatten()
                values = pd.Series(values).dropna()

                if len(values) == 0:
                    continue

                counts, bins = np.histogram(
                    values,
                    bins=self.bins,
                    range=(-self.xlim_resol, self.xlim_resol)
                )

                if bottom is None:
                    bottom = np.zeros_like(counts)

                plt.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom,
                    color=self.get_sample_color(item["sample"]),
                    edgecolor="black",
                    label=item["sample"],
                    align="edge"
                )

                bottom += counts

            contr, recon = key.rsplit("_", 1)

            plt.title(f"Resolution: {recon} vs {contr}")
            plt.xlabel(recon)
            plt.ylabel("Events")
            plt.legend()
            plt.grid(True)

            plt.savefig(
                f"resolution_plots/Resolution_{recon}_from_{contr}.png",
                dpi=300
            )
            plt.close()


### MAIN
if __name__ == "__main__":
    plotter = Plotter(100, 50, 20, 1)
    plotter.load_data()
    plotter.apply_selection()
    plotter.set_parameters()
    plotter.batch(batch_size=250)
    plotter.control_plot()
    plotter.resolution_plot()