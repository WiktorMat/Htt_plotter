from __future__ import annotations

from pathlib import Path
import logging

from htt_plotter.io.hist_parquet import read_histograms_parquet
from htt_plotter.plotting.render import (
    save_stacked_plot,
    save_data_mc_ratio_plot,
)
from htt_plotter.config.loader import load_configs


class ParquetRenderer:

    def __init__(
        self,
        base_dir="plots",
        output_suffix=None,
        process_draw_order=None,
        process_colors=None,
    ):
        self.base_dir = Path(base_dir)

        if output_suffix:
            self.base_dir = self.base_dir / output_suffix

        self.process_draw_order = process_draw_order or []
        self.process_colors = process_colors or {}

        self.logger = logging.getLogger(__name__)

    def get_color(self, process: str):
        return self.process_colors.get(process, "#999999")
    
    def _folder(self, name: str) -> Path:
        path = self.base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def order_mapping_by_list(self, hist: dict, order: list):
        if not order:
            return hist
        return {k: hist[k] for k in order if k in hist}

    def _render_stacked(self, folder: Path, title_prefix: str):
        for file in folder.glob("*.parquet"):
            parsed = read_histograms_parquet(file)
            if parsed is None:
                continue

            _, variable, edges, hist = parsed

            out_path = folder / f"{file.stem}.png"

            save_stacked_plot(
                self.order_mapping_by_list(hist, self.process_draw_order),
                edges,
                title=f"{title_prefix}: {variable}",
                xlabel=variable,
                out_path=str(out_path),
                get_color=self.get_color,
            )

            self.logger.info("Rendered: %s", out_path)

    def render_control(self):
        self._render_stacked(
            self._folder("control_plots"),
            "Control",
        )

    def render_resolution(self):
        self._render_stacked(
            self._folder("resolution_plots"),
            "Resolution",
        )

    def render_mc_data(self):
        folder = self._folder("mc_data_plots")

        for file in folder.glob("*.parquet"):
            parsed = read_histograms_parquet(file)
            if parsed is None:
                continue

            _, variable, edges, hist = parsed

            data_counts = None
            mc_samples = {}

            mc_samples = {}

            for sample, counts in hist.items():
                if sample.lower() == "data":
                    data_counts = counts
                else:
                    mc_samples[sample] = counts

            mc_samples = {
                k: mc_samples[k]
                for k in self.process_draw_order
                if k in mc_samples
            }

            if data_counts is None:
                continue

            out_dir = folder
            out_dir.mkdir(exist_ok=True)

            out_path = out_dir / f"{file.stem}.png"

            save_data_mc_ratio_plot(
                bin_edges=edges,
                data_counts=data_counts,
                mc_samples=mc_samples,
                out_path=str(out_path),
                xlabel=variable,
                get_color=self.get_color,
            )

            self.logger.info("Rendered MC/Data: %s", out_path)

    def run_all(self, do_control=True, do_resolution=True, do_mc_data=True):

        if do_control:
            self.render_control()

        if do_resolution:
            self.render_resolution()

        if do_mc_data:
            self.render_mc_data()