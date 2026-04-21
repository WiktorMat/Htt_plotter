from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
        
        self.process_config = self.process_config

        self.sample_to_process = self._build_sample_to_process_map()
        self.process_colors = self._build_process_colors()
        print("PROCESS COLORS:", self.process_colors)
        print("SAMPLE MAP:", self.sample_to_process)

        runtime = dict(self.plotter_config.get("plotter_runtime") or {})

        if xlim_control is not None:
            runtime["xlim_control"] = xlim_control
        if xlim_resolution is not None:
            runtime["xlim_resolution"] = xlim_resolution
        if bins is not None:
            runtime["bins"] = bins
        if alpha is not None:
            runtime["alpha"] = alpha
        if layout is not None:
            runtime["layout"] = layout
        if mode is not None:
            runtime["mode"] = mode

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

        self.logger = logging.getLogger(__name__)

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
        return self.sample_to_process.get(sample, sample)

    def _get_process_color(self, process: str) -> str:
        return self.process_colors.get(process, "black")

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

    @staticmethod
    def _to_numpy(batch, name: str) -> np.ndarray:
        # Assumes numeric columns; works well for typical parquet physics tables.
        col = batch.column(batch.schema.get_field_index(name))
        return col.to_numpy(zero_copy_only=False)
    


    def _save_histograms_parquet(self, histograms: dict[str, np.ndarray], edges: np.ndarray, out_path: str, plot_type: str, variable: str,) -> None:
        if not histograms:
            return

        samples = []
        plot_types = []
        variables = []
        counts_data = []
        edges_data = []

        edges_list = edges.astype(float).tolist()

        for sample, counts in histograms.items():
            samples.append(sample)
            plot_types.append(plot_type)
            variables.append(variable)
            counts_data.append(counts.astype(float).tolist())
            edges_data.append(edges_list)

        table = pa.Table.from_arrays(
            [
                pa.array(plot_types, type=pa.string()),
                pa.array(variables, type=pa.string()),
                pa.array(samples, type=pa.string()),
                pa.array(counts_data, type=pa.list_(pa.float64())),
                pa.array(edges_data, type=pa.list_(pa.float64())),
            ],
            names=[
                "plot_type",
                "variable",
                "sample",
                "counts",
                "bin_edges",
            ],
        )

        parquet_path = Path(out_path).with_suffix(".parquet")
        pq.write_table(table, parquet_path, compression="zstd")

        self.logger.info("Saved histogram parquet: %s", parquet_path)

    def run_from_histograms(self, *, do_control=True, do_resolution=True, do_mc_data=True): 
        self.logger.info("Running in HIST mode")

        if do_control:
            folder = Path("plots/control_plots")

            for file in folder.glob("*.parquet"):
                table = pq.read_table(file)
                df = table.to_pandas()

                if df.empty:
                    continue

                variable = df["variable"].iloc[0]
                edges = np.array(df["bin_edges"].iloc[0], dtype=float)

                hist = {}
                for _, row in df.iterrows():
                    hist[row["sample"]] = np.array(row["counts"], dtype=float)

                out_path = folder / f"{variable}.png"

                save_stacked_plot(
                    hist,
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
                table = pq.read_table(file)
                df = table.to_pandas()

                if df.empty:
                    continue

                variable = df["variable"].iloc[0]
                edges = np.array(df["bin_edges"].iloc[0], dtype=float)

                hist = {}
                for _, row in df.iterrows():
                    hist[row["sample"]] = np.array(row["counts"], dtype=float)

                out_path = folder / f"{file.stem}.png"

                save_stacked_plot(
                    hist,
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
                table = pq.read_table(file)
                df = table.to_pandas()

                if df.empty:
                    continue

                variable = df["variable"].iloc[0]
                edges = np.array(df["bin_edges"].iloc[0], dtype=float)

                data_counts = None
                mc_samples = {}

                for _, row in df.iterrows():
                    sample = row["sample"]
                    counts = np.array(row["counts"], dtype=float)

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
                    mc_samples=mc_samples,
                    out_path=str(out_path),
                    xlabel=variable,
                    get_color=self._get_process_color,
                )

                self.logger.info("Rendered mc/data from parquet: %s", out_path)
    
    def run_all(self, *, do_control: bool = True, do_resolution: bool = True, do_mc_data: bool = True) -> None:
        """Fill and render all plots in a single pass over input data."""        

        if self.mode == "hist":
            self.run_from_histograms(
                do_control=do_control,
                do_resolution=do_resolution,
                do_mc_data=do_mc_data,
            )
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
            process_kinds: dict[str, str] = {}

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

                self.logger.debug("Columns BEFORE (%s): %s", sample, columns)
                columns = [c for c in columns if c in item["schema"]]
                self.logger.debug("Columns AFTER  (%s): %s", sample, columns)

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

                                process = self._sample_to_process(sample)

                                region = agreement[var][region_name]

                                if process not in region:
                                    region[process] = {
                                        "counts": np.zeros(len(edges) - 1, dtype=float),
                                        "sumw2": np.zeros(len(edges) - 1, dtype=float),
                                    }

                                region[process]["counts"] += counts
                                region[process]["sumw2"] += sumw2

            if do_mc_data:
                self.logger.info("Process kinds (for MC/Data): %s", process_kinds)

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
                        get_color=self._get_process_color,
                        layout=self.layout,
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
                        get_color=self._get_process_color,
                        layout=self.layout,
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
                        process_kinds,
                    )

                    samples = histograms["OS"]

                    data_counts = None
                    mc_samples: dict[str, np.ndarray] = {}

                    for name, hist in samples.items():
                        if process_kinds.get(name, "mc") == "data":
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
                        get_color=self._get_process_color,
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
        else:
            self.logger.warning("There is no mode like that!")


    # Backward-compatible convenience methods
    def control_plot(self) -> None:
        self.run_all(do_control=True, do_resolution=False, do_mc_data=False)

    def resolution_plot(self) -> None:
        self.run_all(do_control=False, do_resolution=True, do_mc_data=False)

    def Plot_MC_Data_Agrement(self) -> None:
        self.run_all(do_control=False, do_resolution=False, do_mc_data=True)
