# Htt_plotter Documentation

# 1. Overview
    `Htt_plotter` is a Python-based plotting framework used to generate histograms from structured datasets such as CSV and Parquet files.

    The program reads user-defined configurations stored in the `Configurations/` directory and produces:
    - control histograms,
    - resolution histograms,
    - comparisons between Monte Carlo (MC) simulation and real data,
    - stacked plots,
    - Data/MC ratio plots,
    - QCD background estimations.

    The core component of the project is the `Plotter` class, which manages the full workflow: loading data, applying selections, filling histograms, and saving plots.

# 2. Main Features
    Users can control:
    - which dataset is used,
    - how data is filtered,
    - which variables are plotted,
    - histogram appearance,
    - plot layout,
    - binning and axis ranges,
    - output configuration.

    The system is modular, so each file is responsible for a specific part of the pipeline.

# 3. Project Structure

## `plotter.py`

    The Plotter class is the core engine of the Htt_plotter framework.
    It is responsible for the full analysis pipeline: from raw input data → event selection → histogram filling → physics plots → saved outputs.
    The class is designed to be:
        - memory efficient (streaming-based processing),
        - config-driven (no hardcoded physics logic),
        - modular (control / resolution / MC vs Data / QCD separation),
        - production-ready for HEP workflows.
    
    Overall Pipeline

        When Plotter.run_all() is executed, the workflow is:
        1. Configuration loading
            Reads all configuration files from: Configurations/<config_name>/
            Loads:
                - datasets (files.json),
                - process mapping (process.json),
                - variable definitions,
                - plotting settings,
                - selection cuts;
                
        2. Dataset indexing (load_index)
            Each dataset sample is indexed once:
                - file locations are resolved,
                - schema (available columns) is detected,
                - metadata (MC / Data / scale factors) is attached,

            This avoids re-reading file metadata during runtime.

        3. Variable setup (set_parameters)
            Defines what will be plotted:

                Control variables - Direct physics distributions (e.g. pT, η, mass),
                Resolution pairs - Two-variable comparisons: reconstructed vs generated,

            Only variables that exist in dataset schema are used.

        4. Binning system (_bin_edges)
            - Bin edges are computed per variable
            - Controlled by:
                - number of bins,
                - x-axis ranges,
                - variable-specific configuration;

            - Binning is cached for performance.

        5. Event loop (core processing)
            Data is processed batch-by-batch using streaming (PyArrow).

            For each sample:
                - selection cuts are applied,
                - weights (MC normalization) are computed,
                - histograms are filled incrementally;

        6. Histogram types
            The framework produces 3 main plot categories:
                a) Control plots:
                    One-dimensional distributions:
                        - filled per physics process,
                        - stacked MC contributions,
                        - optionally compared with Data;
                b) Resolution plots:
                    Two variables comparisions:
                        resolution which is (r - c)/c (where c - control value, r - reconstructed value)
                c) MC vs Data plots:
                    For validation and physics agreement studies:
                        - Data is compared with MonteCarlo predictions,
                        - Includes:
                            1) stacked MC histograms,
                            2) data points,
                            3) radio plot (Data/MC);
        
        7. MC weighting system
            For MC samples:
                - event weights are computed using: compute_mc_weight(),
                - cached per sample for efficiency;

            Used in:
                - MC/Data comparisons,
                - normalization,
                - systematic scaling;

        8. QCD estimation (SS method)
            QCD background is estimated using Same-Sign (SS) technique:
                - Workflow:
                    a) Build SS region template,
                    b) Subtract MC contributions,
                    c) Scale to OS region;
                - Handled internally via: add_qcd_from_ss()

        9. Output system
            All outputs are written automatically:
                a) Plots:
                    plots/control_plots/
                    plots/resolution_plots/
                    plots/mc_data_plots/
                b) Data format:
                    PNG (visualization),
                    Parquet (histogram storage for replotting);

        10. Histogram persistence
            Each plot can optionally be saved as structured data:
                - bin edges,
                - process histograms,
                - MC / Data separation;

            This allows:
                - re-rendering without recomputation,
                - fast iteration on plotting style;
                
        11. Runtime modes
            The plotter supports two execution modes:
                1) mode "raw", full pipeline:
                    - reads parquet input,
                    - applies selections,
                    - fills histograms,
                    - produces plots;
                    --recommended for production
                2) mode "hist", only rendering:
                    - loads precomputed histograms from parquet,
                    - no event loop,
                    - fast replotting;
                    --useful for styling iterations
        12. Key design principles
            1) Single-pass processing - All histograms are filled in one streaming loop.
            2) Schema-aware execution - Only available variables are processed.
            3) Process abstraction layer - Samples are grouped into physics processes: 
                - DY,
                - TT,
                - W+jets,
                - Data,
                - Others;
            4) Memory efficiency - No full dataset loading required.
            5) Config-driven system - No physics logic hardcoded in codebase.

        13. Summary
            The Plotter class is a full analysis engine that: transforms raw HEP datasets into validated physics plots in a single automated pipeline.
            It handles:
                - data ingestion,
                - selection cuts,
                - MC weighting,
                - histogram production,
                - QCD estimation,
                - plotting + persistence;

## `data_acces.py`
    The DataAccess class is the I/O layer of Htt_plotter.
    It is responsible for:
        - locating input files,
        - detecting file formats,
        - reading data efficiently,
        - streaming data in batches to the Plotter;

    Purpose - DataAccess abstracts all file handling so that the Plotter:
        - does not need to know where data comes from,
        - always receives data in a consistent format,
        - can operate on large datasets without loading everything into memory;

    Supported input formats - Parquet, CSV;

    1. Dataset Indexing (build_index)
        This step builds a structured representation of all datasets.
        Each sample is converted into an index entry:
            {
                "sample": str,
                "kind": "mc" | "data",
                "scale": float,
                "color": str | None,
                "files": list[Path],
                "format": "parquet" | "csv",
                "schema": set[str],
            }
        
        Automatic optimizations
            Merged file detection
                If both exist:
                    - merged.parquet,
                    - many job_*.parquet;
                only merged.parquet is used(to avoid double counting and reduce I/O)

        Schema detection
            schema(column names) is read once per sample,
            not per file → improves performance significantly;
        
        Schema caching
            To avoid repeated remote reads: 
                - schema is cached locally in: .cache/htt_plotter/schema/
            cache is reused if:
                - file count is unchanged,
                - file metadata (size, timestamp) matches;

    2. File resolution
        Paths from config are:
            - converted to absolute paths,
            - deduplicated,
            - validated(non-existing files are ignored);

    3. Batch processing(iter_batches)
        This is the core data streaming mechanism.
        Instead of loading full datasets, data is processed in chunks:
            for batch in data_access.iter_batches(...):
                process(batch)
        Parquet processing
            Uses:
                - pyarrow.dataset,
                - multithreaded scanning,
                - streaming batches;

            Features:
                - column selection,
                - filtering at I/O level,
                - configurable batch size,
                - chunking for large file collections;

        Performance features
            Dataset caching:
                - avoids rebuilding dataset objects,
                - limited by max_cached_datasets;
            File chunking - For very large datasets:
                splits file list into smaller groups,
                prevents memory issues;
        
        CSV processing
                - uses pandas.read_csv,
                - converts to Arrow batches,
            Limitations:
                - slower,
                - no native filtering,
                - higher memory usage;

    4. Filtering
        Selections are applied at the I/O level using: make_arrow_filter(...)
        Benefits:
        - reduces data read from disk,
        - improves performance,
        - avoids unnecessary memory usage;
    
    5. Column selection
        Only required columns are read:
            - control variables,
            - resolution variables,
            - selection variables;

    6. Performance design
        The system is optimized for large-scale datasets:
            Streaming - No full dataset loading,
            Single schema read - One per sample, not per file,
            Minimal I/O - Only required columns are loaded,
            Parallel processing - PyArrow uses multithreading,
            Smart caching: schema cache, dataset cache;

    7. Summary
        DataAccess is a high-performance data layer that:
            - abstracts file handling,
            - optimizes data reading,
            - streams data efficiently,
            - enables the Plotter to scale to large datasets;

## `loader.py`
    The loader.py module is responsible for loading and validating all configuration files used by Htt_plotter.
    It acts as the central configuration entry point for the entire pipeline.

    Purpose - The load_configs() function:
        - loads all required configuration files,
        - resolves their locations,
        - applies defaults,
        - validates structure,
        - returns ready-to-use configuration objects for the Plotter;
    
    1. Configuration structure
        All configurations are expected in: Configurations/<config_name>/
        Required files:
            - files.json,
            - process.json,
            - params.yaml,
            - variables.json,
            - plotter.yaml;

    2. Fallback mechanism
        If a file is missing in the selected config: Configurations/<config_name>/
        it is automatically loaded from: source/
    
    3. Main function
        load_configs(project_root, config_name) - Returns:
            (
                sample_config,
                params,
                variable_config,
                plotter_config,
                process_config
            )

        Returned objects
            sample_config:
                - dataset definitions,
                - extracted from files.json,
                - supports both formats:
                    a - flat dict,
                    b - nested under "samples";
            params:
                - physics parameters - YAML,
                - used for:
                    a - MC normalization,
                    b - cross-sections,
                    c - weights;
            variable_config:
                - variable metadata,
                - includes:
                    a - binning hints
                    b - variable types (e.g. angle)
                    c - resolution settings
            plotter_config - Main runtime configuration:
                    a - plotting variables,
                    b - selections,
                    c - runtime settings;

    4. Validation & normalization
        The loader automatically cleans and validates configuration content.
        Resolution variables validation
            Only valid pairs are accepted: ["control", "reconstruction"],
            Invalid entries are skipped: "pt" or ["pt"];

    5. Summary
        loader.py is the configuration backbone of the system.
        It ensures that:
            - all configs are loaded consistently,
            - missing files fall back to defaults,
            - data structures are validated,
            - the Plotter receives clean, ready-to-use inputs;

## `render.py`
    The render.py module is responsible for visualizing and saving all plots generated by Htt_plotter.
    It converts histogram data into publication-style figures using Matplotlib.

    Purpose - This module handles:
        - plot styling and layout,
        - stacking and overlay logic,
        - legends and labels,
        - uncertainty visualization,
        - saving plots to disk;

    1. Main functions
        save_stacked_plot(...)
        Used for:
            - control plots,
            - resolution plots;
        
        Inputs:
            - histograms - Dictionary: {"process_1": np.array([...]),"process_2": np.array([...])},
            - edges - Bin edges (NumPy array),
            - title, xlabel - Plot labels,
            - out_path - Output file path,
            - get_color - Function mapping process to get color,
            - alpha - Transparency,
            - layout - Plot style: [stacked, overlay, side_by_side],
    
    2. Plot layouts
        a. Stacked:
            - histograms stacked on top of each other,
            - represent total MC contribution;
            --recommended for physics plots
        b. Overlay:
            - histograms drawn on top of each other,
            - semi-transparent;
            --useful for shape comparison
        c. Side_by_side 
            - separate subplot per process;
            --useful for debugging or inspection

    3. Plot features
        All plots include:
            - bin-centered bars,
            - black edges for visibility,
            - legend,
            - axis labels,
            - title,
            - tight layout for clean output;

    4. Output
        Plots are saved as: plt.savefig(out_path);
        Typically:
            - PNG format,
            - stored in plots/ directories;

    5. MC vs Data plotting
        save_data_mc_ratio_plot(...)
        Used for:
            - Data vs MC comparisons,
            - validation plots,
            - final physics outputs;
        Structure of the plot. The figure consists of two panels:
            a. Top panel:
                - stacked MC histograms,
                - data points,
                - MC uncertainty band;
            b. Bottom panel (ratio = Data/MC):
                - Data / MC ratio,
                - uncertainty bands,
                - reference line at 1.0;
    
    6. Inputs
        - bin_edges - Histogram bin edges,
        - data_counts - Data histogram,
        - mc_samples - Dictionary of MC histograms,
        - data_unc - Data uncertainties,
        - mc_total_unc - MC uncertainties,
        - xlabel - X-axis label,
        - out_path - Output file;

    7. Uncertainty handling
        If not provided:
            - Data uncertainty: sqrt(N),
            - MC uncertainty: sqrt(N);
        Visualization
            - MC uncertainty - hatched band,
            - Data - error bars,
            - Ratio uncertainty - propagated;

    8. Special handling (QCD)
        - QCD is rendered in gray,
        - labeled as: QCD (from SS);
    
    9. Plot styling details
        - consistent binning,
        - shared X-axis (main + ratio panel),
        - automatic scaling,
        - ratio range fixed to: [0.5, 1.5];

    10. Summary
        render.py is responsible for turning histogram data into final physics plots.
        It provides:
            - flexible layouts(stacked / overlay / side-by-side),
            - high-quality Data vs MC comparisons,
            - uncertainty visualization,
            - consistent styling across all outputs;

## `selection.py`
    The selection.py module is responsible for filtering datasets before plotting.
    It applies physics-based selection cuts to ensure that only relevant events are processed.

    Purpose - This module handles:
        - applying selection criteria,
        - determining required columns,
        - building efficient filters for large datasets,
        - ensuring consistency of selection across the pipeline;

    1. Selection configuration
        Selection cuts are defined in:
            Configurations/<config_name>/plotter.yaml;

        Example:
            selection:
                pt_1_min: 25
                pt_2_min: 20
                eta_1_abs_max: 2.3
                iso_1_max: 0.15;

    2. Supported selection cuts
        - pt_1_min → pt_1 > value,
        - pt_2_min → pt_2 > value,
        - eta_1_abs_max → |eta_1| < value,
        - eta_2_abs_max → |eta_2| < value,
        - decayModePNet_2_eq → exact match,
        - idDeepTau2018v2p5VSjet_2_min → ≥ value,
        - idDeepTau2018v2p5VSe_2_min → ≥ value,
        - idDeepTau2018v2p5VSmu_2_min → ≥ value,
        - iso_1_max → < value,
        - ip_LengthSig_1_abs_min → |value| > threshold;

    3. Main functions
        selection_columns_used(...) - Used for:
            - identifying which columns are needed for selection,
            - minimizing data loading;

        selection_mask(...) - Used for:
            - creating a boolean mask for pandas DataFrames,
            - applying all selection cuts at once;

        make_selector(...) - Used for:
            - generating a reusable selection function,
            - applying selection in modular workflows;

        make_arrow_filter(...) - Used for:
            - building a pyarrow filter expression,
            - filtering data before loading into memory;
            --critical for performance on large datasets

        SELECT(...) - Used for:
            - quick selection on a DataFrame or parquet file,
            - simple standalone usage;

        plotting_columns(...) - Used for:
            - determining available columns for plotting,
            - separating control and resolution variables;

    4. Filtering modes
        a. Pandas filtering:
            - uses selection_mask,
            - operates on in-memory DataFrames;

        b. PyArrow filtering:
            - uses make_arrow_filter,
            - applied during file reading,
            - reduces I/O and memory usage;
            --recommended for large datasets

    5. Behavior
        - all selection conditions are combined using logical AND,
        - missing columns are ignored safely,
        - if no selection is defined → all events are used;

    6. Performance features
        - loads only necessary columns,
        - supports early filtering (before data loading),
        - compatible with batch processing;

    7. Summary
        selection.py is responsible for filtering input data before plotting.
        It ensures:
            - consistent selection logic,
            - efficient data processing,
            - compatibility with both pandas and pyarrow workflows;

## `qcd.py`
    The qcd.py module is responsible for estimating the QCD background using the SS (Same-Sign) method.
    It derives the QCD contribution directly from data and propagates it to the OS (Opposite-Sign) region.

    Purpose - This module handles:
        - QCD background estimation,
        - subtraction of MC from data,
        - propagation of QCD from SS to OS,
        - uncertainty calculation;

    1. Main function
        add_qcd_from_ss(...)
        Used for:
            - estimating QCD contribution from SS region,
            - adding QCD as an additional process to histograms;

    2. Method (SS → OS)
        The QCD estimation follows: QCD (SS) = Data (SS) - MC (SS)
        Then propagated to OS: QCD (OS) = QCD (SS) × qcd_ff

        where:
            - SS = Same-Sign region,
            - OS = Opposite-Sign region,
            - qcd_ff = transfer (fake factor);

    3. Inputs
        - histograms:
            Structure:
                {
                    "SS": {sample: {"counts": ..., "sumw2": ...}},
                    "OS": {sample: {"counts": ..., "sumw2": ...}}
                }

        - config:
            {
                "add_qcd_from_ss": True/False,
                "qcd_ff": float
            }

        - sample_kinds:
            Dictionary mapping sample → type:
                - "data"
                - "mc"

    4. Workflow
        Step 1: identify data samples in SS region;

        Step 2: sum all data histograms;

        Step 3: sum all MC histograms (excluding QCD);

        Step 4: compute QCD in SS: QCD = Data - MC;

        Step 5: compute uncertainties: sumw2 = data_sumw2 + mc_sumw2;

        Step 6: store QCD in SS region;

        Step 7: scale QCD to OS using qcd_ff;

    5. Output
        The function modifies input histograms in-place:
            SS["QCD"] → raw QCD estimate  
            OS["QCD"] → scaled QCD estimate  

    6. Behavior and safeguards
        - if add_qcd_from_ss is False → function does nothing,
        - if no data samples are found → warning is printed,
        - if no MC samples are found → warning is printed,
        - QCD sample is excluded from MC subtraction;

    7. Uncertainty handling
        - uncertainties are propagated as: sumw2 = sumw2_data + sumw2_mc,
        - scaling to OS: sumw2_OS = sumw2_SS × (qcd_ff)^2;

    8. Notes
        - no additional normalization between SS and OS is applied,
        - method assumes SS region is QCD-dominated,
        - widely used in physics analyses for data-driven background estimation;

    9. Summary
        qcd.py provides a simple and robust way to estimate QCD background.
        It:
            - derives QCD directly from data,
            - avoids MC modeling of QCD,
            - integrates seamlessly with MC/Data comparison plots.

# 4. Configurations Directory
    All user configurations are stored in folder:
        Configurations/

# 5. Utilities (Scripts)

## `json_generator.py`

    This script automatically generates configuration files used by Htt_plotter, including:

    - files.json – dataset file definitions
    - process.json – physics process grouping and styling

    It scans the MC output directories and builds a structured configuration for the plotting pipeline.

### Main Functions:
    - get_color() - Assigns a color from a predefined palette to each process.
    - smart_group() - Groups samples into physics categories (e.g. DY, tt, W+jets, data).
    - scan_mc_samples() - Scans MC directories and collects available .parquet files.
    - build_process_map() - Groups samples into physics processes.
    - add_process_colors() - Assigns colors to each process.
    - write_json() - Saves output configuration files.
    - main() - Full pipeline: scans data → builds process map → writes JSON configs.

### Example usage:
    python json_generator.py \
  --config-name config_0 \
  --year Run3_2024 \
  --channel mt \
  --mc-base data/output/test_plotter \
  --path-mode auto

## `merge_parquet.py`
    This script merges multiple .parquet files into a single optimized file using PyArrow streaming.
    It is designed for large-scale HEP datasets where loading everything into memory is not feasible.

### Main Functions:
    - collect_files() - Recursively collects all .parquet files from input directory.
    - merge_parquet() - Streams data in batches and writes a single output file using ParquetWriter.
    - main() - Parses CLI arguments and runs the merge process.

### Example usage:
    python merge_parquet.py \
  --input data/input/samples \
  --output data/output/merged.parquet \
  --batch-size 1000

## `plot_3D.py`:
    This script generates a full 3D event visualization of particle interactions based on reconstructed physics objects stored in .parquet files.
    It visualizes detector geometry, particle tracks, and decay chains in 3D space, including taus, muons, and pion decay products.

### Main Functions:
    - Loads MC datasets from processed parquet files,
    - Maps samples to physics processes using process.json,
    - Converts (pt, eta, phi) → 3D momentum vectors,
    - Reconstructs decay topology of tau leptons,
    - Renders detector geometry (cylindrical approximation),
    - Draws particle tracks with optional glow effect,
    - Differentiates particle types using color coding:
        - Tau: white
        - Muon: cyan
        - Charged pions (π±): lime
        - Neutral pions (π0): magenta

### Example usage:
    Set `--mode 3D_Plot` to run the 3D visualization pipeline.

## 6. Main Execution Script
### `run.py`
    This script is the main entry point for running the full Htt_plotter pipeline.
    It is responsible for:
    - loading configuration files (plotter.yaml),
    - validating config paths,
    - initializing the Plotter,
    - optionally running the ParquetRenderer,
    - executing the full plotting workflow.
    - Workflow:
    - Parse command-line arguments,
    Load YAML configuration from:
        - Configurations/<config_name>/plotter.yaml
        - Extract runtime parameters,
        - Compute histogram binning strategy,
        - Initialize Plotter,
        - Run pipeline depending on selected mode.
        - CLI Arguments (run.py)
        Required:
            --config – name of configuration directory (e.g. config_0)
        Optional:
            --output – suffix added to output plots/files
            --mode – pipeline execution mode:
        Modes:
            - raw – direct plotting from data
            - hist – histogram-based processing
            - render – post-processing from parquet files
            - 3D_Plot – 3D visualization mode
            Example usage:
                - python run.py --config config_0 --mode raw --output test_run

## 7. Plotter Configuration (plotter.yaml)
    The configuration file controls data selection, histogram behavior, plotting style, and optional advanced features.

### 7.1 Plotting Section
    plotting.control
    Defines control variables (truth-level or reference variables) used in plots.
    Example:
        plotting:
        control:
            - trueMETx
            - trueMETy
    These are used as baseline distributions
    Typically represent true / generated physics quantities
    plotting.resolution
    Defines pairs of variables used for resolution plots:
        plotting:
        resolution:
            - [trueMETx, METx]
            - [trueMETy, METy]

        Each pair means:
            - resolution = reconstructed - true

    Used for:
        - resolution histograms
        - performance validation
        - detector response studies

### 7.2 Plotter Runtime Configuration
    This section controls global plotting behavior and histogram construction.
    plotter_runtime:
        - xlim_control - Defines x-axis range for control plots: xlim_control: 100 - symmetric range: [-100, 100] - used for control histograms
        - xlim_resolution - Defines range for resolution plots: xlim_resolution: [-2, 2], controls visible resolution spread, typically narrow around 0
        - bin_width_resolution - Defines histogram bin width for resolution plots: bin_width_resolution: 0.05, Used to dynamically compute number of bins:
            - bins = (xmax - xmin) / bin_width_resolution
            - bins (fallback option)
            - bins: 20 - used if bin_width_resolution is not defined, static bin count fallback
        alpha - alpha: 0.4, transparency of histograms, affects overlay readability
        layout - Defines plot arrangement style: layout: stacked
        Available options:
            - stacked – histograms stacked vertically
            - overlay – all histograms in one plot
            - side by side – separated panels
        mode - Defines internal runtime mode: mode: raw, Used internally by Plotter.
            Possible values:
                - raw – direct input data processing
                - hist – histogram pipeline mode
                - render – rendering from intermediate parquet
                - 3D_Plot – 3D visualization mode

### 7.3 Extra Plots (Optional Module)
    This section enables additional analysis plots beyond standard pipeline output.
    extra_plots:
        enable: false
        extra_plots.enable
        enable: false
    master switch for extra analysis modules
    if false, entire section is ignored
    
#### 7.3.1 Asymmetry Analysis
    Optional sub-module for angular or CP-related studies.
    asymmetry:
        - enable - enable: false, enables asymmetry computation plots
        - column - column: aco_mu_rho, defines input variable for asymmetry study, must exist in dataset
        - bins - bins: 8, number of bins in angular distribution
        - range - range: [0, 6.283185307179586], full angular range (0 → 2π), used for periodic observables
        - cp_weights - cp_weights: true, enables CP-weighted distributions, used for physics asymmetry studies
        - out_dir - out_dir: plots/extra_plots, output directory for additional plots, independent from main pipeline output
    Summary

    This configuration system controls:
        1. Data selection - control variables, resolution pairs
        2. Histogram construction - binning strategy, axis limits, resolution granularity
        3. Plot appearance - transparency (alpha), layout style
        4. Pipeline behavior - execution mode (raw / hist / render / 3D_Plot)
        5. Advanced physics modules - asymmetry analysis, CP-weighted observables, extra diagnostic plots