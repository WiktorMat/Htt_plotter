from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from htt_plotter.backgrounds.qcd import ensure_qcd_placeholder
from htt_plotter.config.loader import load_configs
from htt_plotter.core.draw_order import order_mapping_by_list, process_draw_order
from htt_plotter.core.sample_processing import merge_sample_result, process_sample
from htt_plotter.io.data_access import DataAccess
from htt_plotter.io.hist_parquet import read_histograms_parquet, write_histograms_parquet
from htt_plotter.plotting.binning import get_binning
from htt_plotter.plotting.pairs import make_resolution_pairs
from htt_plotter.plotting.pipelines import (
    render_control_plots,
    render_mc_data_plots,
    render_resolution_plots,
)
from htt_plotter.plotting.render import save_data_mc_ratio_plot, save_stacked_plot
from htt_plotter.plotting.asymmetry import save_asymmetry_plot
from htt_plotter.utils.fs import ensure_dir
from rich.console import Console
from rich.status import Status
# from tools.Plot_3D import plot_events

try:
    import uproot
except ImportError:
    uproot = None
    print("Warning: uproot not installed, .root files will not be supported.")


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
        config_name: str = "config_0",
        *,
        xlim_control: float | None = None,
        xlim_resolution: float | None = None,
        bins: int | None = None,
        alpha: float | None = None,
        layout: str | None = None,
        mode: str | None = None,
    ):
        self.config_name = config_name
        self.project_root = Path(__file__).resolve().parents[3]

        self.index: list[dict] = []

        self._bin_cache: dict = {}
        self._mc_weight_cache: dict = {}

        self.contr_name: list[str] = []
        self.recon_name: list[str] = []
        self.resolution_pairs: list[tuple[str, str]] = []

        (
            self.sample_config,
            self.params,
            self.variable_config,
            self.plotter_config,
            self.process_config,
        ) = load_configs(self.project_root, self.config_name)

        self.logger = logging.getLogger(__name__)

        self.sample_to_process = self._build_sample_to_process_map()
        self.process_colors = self._build_process_colors()
        self._process_draw_order = process_draw_order(self.process_config)
        self.logger.debug("Process colors: %s", self.process_colors)
        self.logger.debug("Sample→process map: %s", self.sample_to_process)

        runtime = dict(self.plotter_config.get("plotter_runtime") or {})

        runtime_cfg = self.plotter_config.get("plotter_runtime") or {}

        runtime = {
            "xlim_control": runtime_cfg.get("xlim_control", 100),
            "xlim_resolution": runtime_cfg.get("xlim_resolution", 50),
            "bins": runtime_cfg.get("bins", 20),
            "alpha": runtime_cfg.get("alpha", 1.0),
            "layout": runtime_cfg.get("layout", "stacked"),
            "mode": runtime_cfg.get("mode", "raw"),
        }

        self.xlim_ctrl = runtime.get("xlim_control", 100)
        self.xlim_resol = runtime.get("xlim_resolution", 50)
        self.bins = runtime.get("bins", 20)
        self.alpha = runtime.get("alpha", 1.0)
        self.layout = runtime.get("layout", "stacked")
        self.mode = runtime.get("mode", "raw")

        self.data_access = DataAccess(
            self.project_root,
            self.sample_config,
            log_every_files=200,
        )

    def _build_sample_to_process_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}

        for process_name, cfg in (self.process_config or {}).items():
            if isinstance(cfg, dict):
                samples = cfg.get("samples", [])
            elif isinstance(cfg, list):
                samples = cfg
            else:
                samples = []

            for sample in samples:
                mapping[sample] = process_name

        return mapping


    def _build_process_colors(self) -> dict[str, str]:
        colors: dict[str, str] = {}

        for process_name, cfg in (self.process_config or {}).items():
            if isinstance(cfg, dict):
                colors[process_name] = cfg.get("color", "black")
            else:
                colors[process_name] = "black"

        return colors


    def _sample_to_process(self, sample: str) -> str:
        return self.sample_to_process.get(sample, "unknown")

    def _get_process_color(self, process: str) -> str:
        return self.process_colors.get(process, "#999999")

    def load_index(self) -> None:
        self.index = self.data_access.build_index()
        n_files = sum(len(i.get("files", [])) for i in self.index)
        self.logger.info("Indexed samples: %d | total files: %d", len(self.index), n_files)

    def set_parameters(self) -> None:
        desired_control = (self.plotter_config.get("plotting") or {}).get("control", [])
        desired_resolution = (self.plotter_config.get("plotting") or {}).get("resolution", [])

        available_cols = set()
        for item in self.index:
            available_cols |= set(item.get("schema", set()))

        self.contr_name = [v for v in desired_control if v in available_cols]

        if not self.contr_name:
            self.logger.warning("No valid control vars found → falling back to defaults")
            self.contr_name = list(available_cols)[:2]

        self.resolution_pairs = []

        for item in desired_resolution:

            if isinstance(item, (list, tuple)) and len(item) == 2:
                c, r = item
                if c in available_cols and r in available_cols:
                    self.resolution_pairs.append((c, r))

            elif isinstance(item, str):
                self.logger.warning(
                    "Resolution item '%s' is a string, expected [c,r] pair. Ignored.",
                    item
                )

            else:
                raise ValueError(
                    f"Invalid resolution format: {item}. Expected [var1, var2]."
                )

        self.logger.info("Control vars: %s", self.contr_name)
        self.logger.info("Resolution pairs: %s", self.resolution_pairs)

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

    def _run_all_raw(
        self,
        *,
        do_control: bool,
        do_resolution: bool,
        do_mc_data: bool,
        n_workers: int = 1,
    ) -> None:
        enable_control = bool(do_control and self.contr_name)
        enable_resolution = bool(do_resolution and self.resolution_pairs)
        enable_mc_data = bool(do_mc_data and self.contr_name)

        extra_cfg = self.plotter_config.get("plotter_runtime", {}).get("extra_plots", {})
        asym_cfg = extra_cfg.get("asymmetry", {}) if isinstance(extra_cfg, dict) else {}
        enable_asym = bool(extra_cfg.get("enable", False) and asym_cfg.get("enable", False))

        self.logger.info(
            "Starting run_all: workers=%d | control=%s | resolution=%s | mc_data=%s | asym=%s",
            n_workers,
            enable_control,
            enable_resolution,
            enable_mc_data,
            enable_asym,
        )

        if enable_control:
            ensure_dir("plots/control_plots")
        if enable_resolution:
            ensure_dir("plots/resolution_plots")
        if enable_mc_data:
            ensure_dir("plots/mc_data_plots")

        control_edges: dict[str, np.ndarray] = {}
        if enable_control or enable_mc_data:
            control_edges = {v: self._bin_edges(v) for v in self.contr_name}

        resolution_edges: dict[tuple[str, str], np.ndarray] = {}
        if enable_resolution:
            resolution_edges = {pair: self._bin_edges(pair[1]) for pair in self.resolution_pairs}

        control_hists: dict[str, dict[str, np.ndarray]] | None = None
        resolution_hists: dict[tuple[str, str], dict[str, np.ndarray]] | None = None
        agreement: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] | None = None

        if enable_control:
            control_hists = {v: {} for v in self.contr_name}
        if enable_resolution:
            resolution_hists = {pair: {} for pair in self.resolution_pairs}
        if enable_mc_data:
            agreement = {v: {"OS": {}, "SS": {}} for v in self.contr_name}

        process_kinds: dict[str, str] = {}
        asymmetry_buffer: dict[str, dict[str, list[np.ndarray]]] = {}

        payload_base = {
            "project_root": str(self.project_root),
            "sample_config": self.sample_config,
            "params": self.params,
            "variable_config": self.variable_config,
            "plotter_config": self.plotter_config,
            "sample_to_process": self.sample_to_process,
            "contr_name": self.contr_name,
            "resolution_pairs": [list(x) for x in self.resolution_pairs],
            "control_edges": control_edges,
            "resolution_edges": resolution_edges,
            "do_control": enable_control,
            "do_resolution": enable_resolution,
            "do_mc_data": enable_mc_data,
            "asym_cfg": asym_cfg if enable_asym else {},
        }

        if n_workers > 1:
            self.logger.info("Starting run_all in multiprocessing mode: workers=%d", n_workers)
            payloads: list[dict] = []
            for item in self.index:
                p = dict(payload_base)
                p["item"] = item
                p["progress_interval_s"] = 30.0
                payloads.append(p)

            completed = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_sample, payload) for payload in payloads]
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1

                    sample = result.get("sample", "<unknown>")
                    self.logger.info(
                        "Multiprocessing progress: %d/%d samples merged (sample=%s)",
                        completed,
                        len(payloads),
                        sample,
                    )

                    merge_sample_result(
                        result,
                        control_hists=control_hists,
                        resolution_hists=resolution_hists,
                        agreement=agreement,
                        asymmetry_buffer=asymmetry_buffer if enable_asym else None,
                        process_kinds=process_kinds if enable_mc_data else None,
                    )

        else:
            console = Console()
            with Status("[bold green]Processing samples...", console=console) as status:
                for item in self.index:
                    sample = item["sample"]
                    kind = item.get("kind", "mc")
                    schema = set(item.get("schema", set()))

                    msg = (
                        f"[cyan]Sample:[/cyan] {sample} | "
                        f"[yellow]kind:[/yellow] {kind} | "
                        f"[magenta]files:[/magenta] {len(item.get('files', []))} | "
                        f"[green]cols:[/green] {len(schema)}"
                    )
                    base_status = msg
                    status.update(base_status)

                    def _on_scan_progress(info: dict) -> None:
                        event = info.get("event")
                        if event == "chunk_start":
                            chunk = info.get("chunk")
                            chunks = info.get("chunks")
                            if chunk is not None and chunks is not None:
                                status.update(f"{base_status} | chunk {chunk}/{chunks}")
                            return

                        if event in {"start", "progress", "done"}:
                            batches = info.get("batches", 0)
                            rows = info.get("rows", 0)
                            elapsed_s = float(info.get("elapsed_s", 0.0))
                            io_s = float(info.get("io_s", 0.0))
                            consumer_s = float(info.get("consumer_s", 0.0))
                            chunk = info.get("chunk")
                            chunks = info.get("chunks")
                            chunk_msg = f" | chunk {chunk}/{chunks}" if chunk is not None and chunks is not None else ""
                            status.update(
                                f"{base_status}{chunk_msg} | batches {batches} | rows {rows} | elapsed {elapsed_s:.1f}s | io {io_s:.1f}s | consumer {consumer_s:.1f}s"
                            )

                    p = dict(payload_base)
                    p["item"] = item
                    p["progress_interval_s"] = 10.0
                    result = process_sample(
                        p,
                        progress_callback=_on_scan_progress,
                        mc_weight_cache=self._mc_weight_cache,
                    )
                    merge_sample_result(
                        result,
                        control_hists=control_hists,
                        resolution_hists=resolution_hists,
                        agreement=agreement,
                        asymmetry_buffer=asymmetry_buffer if enable_asym else None,
                        process_kinds=process_kinds if enable_mc_data else None,
                    )

        if enable_mc_data:
            self.logger.info("Process kinds (for MC/Data): %s", process_kinds)

        if enable_control and control_hists is not None:
            render_control_plots(
                control_hists,
                control_edges,
                process_draw_order=self._process_draw_order,
                get_color=self._get_process_color,
                layout=self.layout,
                logger=self.logger,
            )

        if enable_resolution and resolution_hists is not None:
            render_resolution_plots(
                resolution_hists,
                resolution_edges,
                process_draw_order=self._process_draw_order,
                get_color=self._get_process_color,
                layout=self.layout,
                logger=self.logger,
            )

        if enable_mc_data and agreement is not None:
            render_mc_data_plots(
                agreement,
                control_edges,
                process_draw_order=self._process_draw_order,
                process_kinds=process_kinds,
                get_color=self._get_process_color,
                params=self.params,
                logger=self.logger,
            )

        if enable_asym and asymmetry_buffer:
            for column, data in asymmetry_buffer.items():
                if len(data.get("even") or []) == 0 or len(data.get("odd") or []) == 0:
                    continue
                hist_even = np.sum(data["even"], axis=0)
                hist_odd = np.sum(data["odd"], axis=0)
                save_asymmetry_plot(
                    hist_even=hist_even,
                    hist_odd=hist_odd,
                    cfg=asym_cfg,
                    column=column,
                    logger=self.logger,
                )
    
    def run_from_histograms(self, *, do_control=True, do_resolution=True, do_mc_data=True): 
        self.logger.info("Running in HIST mode")

        if do_control:
            folder = Path("plots/control_plots")

            for file in folder.glob("*.parquet"):
                parsed = read_histograms_parquet(file)
                if parsed is None:
                    continue

                _, variable, edges, hist = parsed

                out_path = folder / f"{variable}.png"

                save_stacked_plot(
                    order_mapping_by_list(hist, self._process_draw_order),
                    edges,
                    title=f"Control: {variable}",
                    xlabel=variable,
                    out_path=str(out_path),
                    get_color=self._get_process_color,
                    layout=self.layout,
                )

                self.logger.info("Rendered control from parquet: %s", out_path)

        if do_resolution:
            folder = Path("plots/resolution_plots")

            for file in folder.glob("*.parquet"):
                parsed = read_histograms_parquet(file)
                if parsed is None:
                    continue

                _, variable, edges, hist = parsed

                out_path = folder / f"{file.stem}.png"

                save_stacked_plot(
                    order_mapping_by_list(hist, self._process_draw_order),
                    edges,
                    title=f"Resolution: {variable}",
                    xlabel=variable,
                    out_path=str(out_path),
                    get_color=self._get_process_color,
                    layout=self.layout,
                )

                self.logger.info("Rendered resolution from parquet: %s", out_path)

        if do_mc_data:
            folder = Path("plots/mc_data_plots")

            for file in folder.glob("*.parquet"):
                parsed = read_histograms_parquet(file)
                if parsed is None:
                    continue

                _, variable, edges, hist = parsed

                data_counts = None
                mc_samples: dict[str, np.ndarray] = {}

                for sample, counts in hist.items():
                    if sample.lower() == "data":
                        data_counts = counts
                    else:
                        mc_samples[sample] = counts

                if data_counts is None:
                    continue

                mc_samples = ensure_qcd_placeholder(
                    mc_samples,
                    self._process_draw_order,
                    data_counts,
                )

                out_path = folder / f"{file.stem}.png"

                save_data_mc_ratio_plot(
                    bin_edges=edges,
                    data_counts=data_counts,
                    mc_samples=order_mapping_by_list(mc_samples, self._process_draw_order),
                    out_path=str(out_path),
                    xlabel=variable,
                    get_color=self._get_process_color,
                )

                self.logger.info("Rendered mc/data from parquet: %s", out_path)

    def run_3DPlot(self, n_events: int = 1):
        import pandas as pd

        self.logger.info("Running in 3D Plot mode")

        if not self.index:
            self.load_index()

        events = []
        sample_names = []

        for item in self.index:
            sample = item["sample"]

            for batch in self.data_access.iter_batches(item):
                df = batch.to_pandas()

                if len(df) == 0:
                    continue

                for i in range(min(n_events, len(df))):
                    events.append(df.iloc[i])
                    sample_names.append(sample)

                break

            if len(events) >= n_events:
                break

        if not events:
            self.logger.warning("No events found for display")
            return

        self.logger.info("Collected %d events for display", len(events))

        plot_events(events, sample_names, elev=23.5, azim=67.2,
                    save_path="plots/event_display.png")

    def _run_all_raw_multiprocessing(
        self,
        *,
        do_control: bool,
        do_resolution: bool,
        do_mc_data: bool,
        n_workers: int,
    ) -> None:
        self._run_all_raw(
            do_control=do_control,
            do_resolution=do_resolution,
            do_mc_data=do_mc_data,
            n_workers=max(1, int(n_workers)),
        )
    
    def run_all(self, *, do_control: bool = True, do_resolution: bool = True, do_mc_data: bool = True) -> None:
        """Fill and render all plots in a single pass over input data."""

        if self.mode == "hist":
            self.run_from_histograms(
                do_control=do_control,
                do_resolution=do_resolution,
                do_mc_data=do_mc_data,
            )
            return
        elif self.mode == "3D_Plot":
            self.run_3DPlot()
            return        
        elif self.mode == "raw":
            if not self.index:
                self.load_index()
            if not self.contr_name and not self.recon_name:
                self.set_parameters()
            if do_resolution and not self.resolution_pairs:
                self.batch()

            runtime_cfg = self.plotter_config.get("plotter_runtime", {}) or {}
            processing_backend = str(runtime_cfg.get("processing_backend", "serial")).strip().lower()
            n_workers = int(runtime_cfg.get("n_workers", 1) or 1)

            effective_workers = (
                n_workers
                if processing_backend in {"multiprocessing", "mp", "process"} and n_workers > 1
                else 1
            )

            self._run_all_raw(
                do_control=do_control,
                do_resolution=do_resolution,
                do_mc_data=do_mc_data,
                n_workers=effective_workers,
            )
            return
        else:
            self.logger.warning("There is no mode like that!")
            return