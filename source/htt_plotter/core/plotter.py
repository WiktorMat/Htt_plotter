from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from htt_plotter.backgrounds.qcd import add_qcd_from_ss
from htt_plotter.config.loader import load_configs
from htt_plotter.core.draw_order import order_mapping_by_list, order_mc_samples, process_draw_order
from htt_plotter.io.data_access import DataAccess
from htt_plotter.io.hist_parquet import read_histograms_parquet, write_histograms_parquet
from htt_plotter.physics.weights import compute_mc_weight
from htt_plotter.plotting.accumulate import add_histogram
from htt_plotter.plotting.binning import get_binning
from htt_plotter.plotting.pairs import make_resolution_pairs
from htt_plotter.plotting.render import save_data_mc_ratio_plot, save_stacked_plot
from htt_plotter.selection.selection import make_arrow_filter, selection_columns_used
from htt_plotter.utils.fs import ensure_dir
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.status import Status
from scripts.Plot_3D import plot_events
from htt_plotter.plotting.asymmetry import plot_asymmetry

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
        output_suffix=None
    ):
        self.config_name = config_name
        self.project_root = Path(__file__).resolve().parents[3]

        self.index: list[dict] = []

        self._bin_cache: dict = {}
        self._mc_weight_cache: dict = {}

        self.contr_name: list[str] = []
        self.recon_name: list[str] = []
        self.resolution_pairs: list[tuple[str, str]] = []

        self.output_suffix = output_suffix

        if self.output_suffix:
            self.base_dir = Path("plots") / self.output_suffix
        else:
            self.base_dir = Path("plots")


        configs = load_configs(self.project_root, self.config_name)

        self.sample_config = configs[0] or {}
        self.params = configs[1] or {}
        self.variable_config = configs[2] or {}
        self.plotter_config = configs[3] or {}
        self.process_config = configs[4] or {}

        self.logger = logging.getLogger(__name__)

        self.sample_to_process = self._build_sample_to_process_map()
        self.process_colors = self._build_process_colors()
        self._process_draw_order = process_draw_order(self.process_config)
        self.logger.debug("Process colors: %s", self.process_colors)
        self.logger.debug("Sample→process map: %s", self.sample_to_process)

        runtime = dict(self.plotter_config.get("plotter_runtime") or {})

        runtime_cfg = self.plotter_config.get("plotter_runtime") or {}

        runtime = {
            "xlim_control": xlim_control if xlim_control is not None else runtime_cfg.get("xlim_control", 100),
            "xlim_resolution": xlim_resolution if xlim_resolution is not None else runtime_cfg.get("xlim_resolution", [-2, 2]),
            "bins": bins if bins is not None else runtime_cfg.get("bins", 20),
            "alpha": alpha if alpha is not None else runtime_cfg.get("alpha", 1.0),
            "layout": layout if layout is not None else runtime_cfg.get("layout", "stacked"),
            "mode": mode if mode is not None else runtime_cfg.get("mode", "raw"),
        }
        
        if isinstance(runtime["xlim_resolution"], (int, float)):
            self.xlim_resol = (-runtime["xlim_resolution"], runtime["xlim_resolution"])
        else:
            self.xlim_resol = tuple(runtime["xlim_resolution"])

        self.xlim_ctrl = runtime.get("xlim_control", 100)
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

    def _bin_edges(self, var: str, *, is_resolution: bool = False) -> np.ndarray:
        if is_resolution:
            if isinstance(self.xlim_resol, (int, float)):
                x_min, x_max = -self.xlim_resol, self.xlim_resol
            else:
                x_min, x_max = self.xlim_resol

            nb = self.bins

            edges = np.linspace(x_min, x_max, nb + 1)
            return edges
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
                ensure_dir(self.base_dir / "control_plots")
            if do_resolution:
                ensure_dir(self.base_dir / "resolution_plots")
            if do_mc_data:
                ensure_dir(self.base_dir / "mc_data_plots")

            # Precompute bin edges
            control_edges = {v: self._bin_edges(v) for v in self.contr_name}
            resolution_edges = {pair: self._bin_edges(pair[1], is_resolution=True) for pair in self.resolution_pairs}

            # Histogram containers
            control_hists: dict[str, dict[str, np.ndarray]] = {v: {} for v in self.contr_name}
            resolution_hists: dict[tuple[str, str], dict[str, np.ndarray]] = {pair: {} for pair in self.resolution_pairs}

            # For MC/Data agreement: per var → OS/SS → sample → {counts,sumw2}
            agreement: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {
                v: {"OS": {}, "SS": {}} for v in self.contr_name
            }
            process_kinds: dict[str, str] = {}

            selection_cfg = self.plotter_config.get("selection", {}) or {}
            selection_cols = selection_columns_used(selection_cfg)

            console = Console()

            table = Table(title="Processing samples")
            table.add_column("Index")
            table.add_column("Sample")
            table.add_column("Kind")

            with Live(table, console=console, refresh_per_second=4) as live:
                for i, item in enumerate(self.index):
                    table.add_row(str(i+1), item["sample"], item.get("kind", "mc"))
                    live.update(table)

            console = Console()

            with Status("[bold green]Processing samples...", console=console) as status:

                extra_cfg = self.plotter_config.get("plotter_runtime", {}).get("extra_plots", {})
                asym_cfg = extra_cfg.get("asymmetry", {}) if isinstance(extra_cfg, dict) else {}

                asymmetry_buffer = {}

                for item in self.index:
                    sample = item["sample"]
                    kind = item.get("kind", "mc")
                    scale = item.get("scale", 1.0)
                    schema = set(item.get("schema", set()))

                    runtime_cfg = self.plotter_config.get("plotter_runtime", {}) or {}
                    prefetch_batches = int(runtime_cfg.get("io_prefetch_batches", 4) or 0)
                    if prefetch_batches < 0:
                        prefetch_batches = 0

                    msg = (
                        f"[cyan]Sample:[/cyan] {sample} | "
                        f"[yellow]kind:[/yellow] {kind} | "
                        f"[magenta]files:[/magenta] {len(item.get('files', []))} | "
                        f"[green]cols:[/green] {len(schema)}"
                    )

                    status.update(msg)

                    self.logger.debug(
                        "Sample start: %s | kind=%s | files=%d | schema_cols=%d",
                        sample,
                        kind,
                        len(item.get("files", [])),
                        len(schema)
                    )

                    base_status = msg

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
                            chunk_msg = (
                                f" | chunk {chunk}/{chunks}" if chunk is not None and chunks is not None else ""
                            )

                            status.update(
                                f"{base_status}{chunk_msg} | batches {batches} | rows {rows} | elapsed {elapsed_s:.1f}s | io {io_s:.1f}s | consumer {consumer_s:.1f}s"
                            )

                    process = self._sample_to_process(sample)
                    # If a process contains any data sample, treat the whole process as data.
                    if kind == "data" or process not in process_kinds:
                        process_kinds[process] = kind

                    # Determine which columns we actually need for this sample.
                    present_control = [v for v in self.contr_name if v in schema]
                    present_pairs = [(c, r) for (c, r) in self.resolution_pairs if c in schema and r in schema]

                    has_work = (
                        (do_control and bool(present_control))
                        or (do_resolution and bool(present_pairs))
                        or (do_mc_data and bool(present_control) and ("os" in schema))
                    )

                    needed: set[str] = set()
                    if do_control:
                        needed |= set(present_control)
                    if do_resolution:
                        for c, r in present_pairs:
                            needed.add(c)
                            needed.add(r)
                    if do_mc_data:
                        needed |= set(present_control)
                        for optional_col in ["os", "trg_singlemuon", "trg_mt_cross", "weight"]:
                            if optional_col in schema:
                                needed.add(optional_col)

                    # Columns needed to evaluate selection (for dataset filter).
                    needed |= {c for c in selection_cols if c in schema}

                    columns = sorted(needed)

                    filter_expr = make_arrow_filter(self.plotter_config, schema)

                    # Compute constant MC weight once per sample for agreement plots.
                    if kind == "data":
                        mc_weight = 1.0
                    else:
                        self.params_fallback = {
                            sample: sample.split("-")[0]
                        }
                        params_key = (
                            (self.sample_config.get(sample) or {}).get("params_key")
                            or sample.split("-")[0].split("_")[0]
                        )
                        mc_weight = compute_mc_weight(params_key, self.params, cache=self._mc_weight_cache)

                    self.logger.debug("Columns BEFORE (%s): %s", sample, columns)
                    columns = [c for c in columns if c in item["schema"]]
                    self.logger.debug("Columns AFTER  (%s): %s", sample, columns)

                    for batch in self.data_access.iter_batches(
                        item,
                        columns=columns,
                        filter_expr=filter_expr,
                        progress_callback=_on_scan_progress,
                        progress_interval_s=10.0,
                        prefetch_batches=prefetch_batches,
                    ):
                        # Control plots
                        if do_control and present_control:
                            for var in present_control:
                                values = self._to_numpy(batch, var)
                                mask = np.isfinite(values)
                                if not np.any(mask):
                                    continue

                                edges = control_edges.get(var)

                                if edges is None or len(edges) < 2:
                                    self.logger.warning(f"No valid binning for {var}, skipping")
                                    continue

                                counts, _ = np.histogram(values[mask], bins=edges)
                                counts = counts * scale
                                process = self._sample_to_process(sample)
                                add_histogram(control_hists[var], process, counts)

                        # Resolution plots
                        if do_resolution and present_pairs:
                            for c, r in present_pairs:
                                cv = self._to_numpy(batch, c)
                                rv = self._to_numpy(batch, r)

                                mask = np.isfinite(cv) & np.isfinite(rv) & (cv != 0)
                                if not np.any(mask):
                                    continue

                                var_cfg = self.variable_config.get(r, {})

                                is_angle = var_cfg.get("type") == "angle"
                                relative = var_cfg.get("relative_resolution", True)

                                if is_angle or not relative:
                                    resolution = rv[mask] - cv[mask]

                                    resolution = (resolution + np.pi) % (2 * np.pi) - np.pi
                                else:
                                    resolution = (rv[mask] - cv[mask]) / cv[mask]
                                edges = resolution_edges[(c, r)]
                                counts, _ = np.histogram(resolution, bins=edges)
                                counts = counts * scale
                                process = self._sample_to_process(sample)
                                add_histogram(resolution_hists[(c, r)], process, counts)

                        # MC/Data agreement (OS/SS)
                        has_os_col = "os" in batch.schema.names

                        if do_mc_data and present_control:
                            if has_os_col:
                                os_flag = self._to_numpy(batch, "os")
                            else:
                                os_flag = np.ones(batch.num_rows, dtype=np.int32)

                            for var in present_control:
                                values = self._to_numpy(batch, var)

                                regions_to_fill = {"OS": os_flag == 1}
                                if has_os_col:
                                    regions_to_fill["SS"] = (os_flag == 0)
                                else:
                                    regions_to_fill["SS"] = np.zeros(batch.num_rows, dtype=bool)

                                for region_name, region_mask in regions_to_fill.items():
                                    mask = region_mask & np.isfinite(values)
                                    
                                    has_trg1 = "trg_singlemuon" in batch.schema.names
                                    has_trg2 = "trg_mt_cross" in batch.schema.names
                                    
                                    if has_trg1 or has_trg2:
                                        trg_mask = np.zeros(batch.num_rows, dtype=bool)
                                        if has_trg1:
                                            trg_mask |= (self._to_numpy(batch, "trg_singlemuon") == 1)
                                        if has_trg2:
                                            trg_mask |= (self._to_numpy(batch, "trg_mt_cross") == 1)
                                        mask &= trg_mask

                                    if not np.any(mask):
                                        continue

                                    if "weight" in batch.schema.names:
                                        weights = self._to_numpy(batch, "weight")
                                    else:
                                        weights = np.ones(batch.num_rows, dtype=float)

                                    edges = control_edges[var]
                                    raw_counts, _ = np.histogram(values[mask], bins=edges, weights=weights[mask])
                                    raw_counts = raw_counts.astype(float)
                                    counts = raw_counts * mc_weight
                                    sumw2 = raw_counts * (mc_weight**2)

                                    process = self._sample_to_process(sample)
                                    region = agreement[var][region_name]

                                    if process not in region:
                                        region[process] = {
                                            "counts": np.zeros(len(edges) - 1, dtype=float),
                                            "sumw2": np.zeros(len(edges) - 1, dtype=float),
                                        }

                                    region[process]["counts"] += counts
                                    region[process]["sumw2"] += sumw2

                        if asym_cfg.get("enable", False):
                            column = asym_cfg.get("column")

                            if column in schema:

                                values = self._to_numpy(batch, column)

                                if asym_cfg.get("cp_weights", True):
                                    w_even = self._to_numpy(batch, "wt_cp_sm")
                                    w_odd = self._to_numpy(batch, "wt_cp_ps")
                                else:
                                    w_even = np.ones_like(values)
                                    w_odd = np.ones_like(values)

                                hist_even, _ = np.histogram(
                                    values,
                                    bins=asym_cfg["bins"],
                                    range=tuple(asym_cfg["range"]),
                                    weights=w_even
                                )

                                hist_odd, _ = np.histogram(
                                    values,
                                    bins=asym_cfg["bins"],
                                    range=tuple(asym_cfg["range"]),
                                    weights=w_odd
                                )

                                buf = asymmetry_buffer.setdefault(column, {"even": [], "odd": []})
                                buf["even"].append(hist_even)
                                buf["odd"].append(hist_odd)

                if do_mc_data:
                    self.logger.info("Process kinds (for MC/Data): %s", process_kinds)

                # Render control
                if do_control:
                    control_dir = self.base_dir / "control_plots"
                    for var, hist in control_hists.items():
                        if not hist:
                            continue
                        out_path = str(control_dir / f"{var}.png")
                        save_stacked_plot(
                            order_mapping_by_list(hist, self._process_draw_order),
                            control_edges[var],
                            title=f"Control: {var}",
                            xlabel=var,
                            out_path=out_path,
                            get_color=self._get_process_color,
                            layout=self.layout,
                        )

                        parquet_path = write_histograms_parquet(
                            histograms=hist,
                            edges=control_edges[var],
                            out_path=out_path,
                            plot_type="control",
                            variable=var,
                        )

                        self.logger.info("Saved control plot: %s (parquet: %s)", out_path, parquet_path)

                # Render resolution
                if do_resolution:
                    res_dir = self.base_dir / "resolution_plots"
                    for (c, r), hist in resolution_hists.items():
                        if not hist:
                            continue
                        out_path = str(res_dir / f"Resolution_{r}_from_{c}.png")
                        save_stacked_plot(
                            order_mapping_by_list(hist, self._process_draw_order),
                            resolution_edges[(c, r)],
                            title=f"Resolution: {r} vs {c}",
                            xlabel=f"res_{r}",
                            out_path=out_path,
                            get_color=self._get_process_color,
                            layout=self.layout,
                        )
                        parquet_path = write_histograms_parquet(
                            histograms=hist,
                            edges=resolution_edges[(c, r)],
                            out_path=out_path,
                            plot_type="resolution",
                            variable=f"{r}_from_{c}",
                        )
                        self.logger.info("Saved resolution plot: %s (parquet: %s)", out_path, parquet_path)

                # Render agreement
                if do_mc_data:
                    for var, regions in agreement.items():
                        # add QCD per-var
                        histograms = {"OS": regions["OS"], "SS": regions["SS"]}

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

                        lumi = self.params.get("lumi", None)

                        if lumi is not None and data_os > 0:
                            data_rate = data_os / lumi
                            mc_total = mc_os

                            self.logger.info(
                                "[DEBUG NORM] %s | Data/Lumi = %.6e | Sum(MC weights) = %.6e | Ratio (MC / (Data/Lumi)) = %.3f",
                                var,
                                data_rate,
                                mc_total,
                                mc_total / data_rate if data_rate > 0 else -1,
                            )

                        if data_os > 0:
                            self.logger.info(
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

                        for name, hist in samples.items():
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


                        data_unc = None
                        if data_sumw2 is not None:
                            data_unc = np.sqrt(np.maximum(data_sumw2, 0.0))

                        mc_total_unc = None
                        if mc_sumw2_total is not None:
                            mc_total_unc = np.sqrt(np.maximum(mc_sumw2_total, 0.0))

                        out_path = self.base_dir / "mc_data_plots" / f"MC_vs_Data_{var}.png"
                        save_data_mc_ratio_plot(
                            bin_edges=control_edges[var],
                            data_counts=data_counts,
                            mc_samples=order_mc_samples(
                                mc_samples,
                                desired_order=self._process_draw_order,
                                process_kinds=process_kinds,
                            ),
                            data_unc=data_unc,
                            mc_total_unc=mc_total_unc,
                            out_path=out_path,
                            xlabel=var,
                            get_color=self._get_process_color,
                        )
                        combined = {"data": data_counts, **mc_samples}

                        parquet_path = write_histograms_parquet(
                            histograms=combined,
                            edges=control_edges[var],
                            out_path=out_path,
                            plot_type="mc_data",
                            variable=var,
                        )

                        self.logger.info("Saved MC/Data plot: %s (parquet: %s)", out_path, parquet_path)

            if extra_cfg.get("enable", False) and asymmetry_buffer:
                asym_run_cfg = asym_cfg.copy()
                
                asym_run_cfg["out_dir"] = str(self.base_dir / "extra_plots")

                for column, data in asymmetry_buffer.items():
                    if len(data["even"]) == 0 or len(data["odd"]) == 0:
                        continue

                    hist_even = np.sum(data["even"], axis=0)
                    hist_odd = np.sum(data["odd"], axis=0)

                    plot_asymmetry(
                        hist_even, 
                        hist_odd, 
                        asym_run_cfg, 
                        column,
                        logger=self.logger
                    )
        else:
            self.logger.warning("There is no mode like that!")