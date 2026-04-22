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

    4. 

## `selection.py`
    Responsible for filtering datasets and selecting variables.
    Used before plotting.

### Main functions:
    - **selection_columns_used()**
    - **selection_mask()**
    - **make_selector()**
    - **make_arrow_filter()**
    - **SELECT()**
    - **plotting_columns()**

## `qcd.py`
    Used for QCD background estimation with the SS method.

    Workflow:
    1. Build QCD template in SS region.
    2. Subtract MC background from data.
    3. Scale result to OS region.

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

# 6. Main Execution Script

## `run.py`
    This script is the main entry point for running the full Htt_plotter pipeline.

    It is responsible for:
        - setting up logging,
        - ensuring configuration files exist,
        - generating missing configs if needed,
        - initializing the Plotter,
        - executing the full plotting workflow.
        
    Workflow:
        - Set up logging to both console and file (plotter.log),
        - Select configuration directory (config_name),
        - Ensure files.json exists (auto-generate if missing),
        - Initialize Plotter with user-defined parameters,
        - Run full pipeline using run_all();

    To run the program with custom parameters, set the following parameters:
        - xlim_control – X-axis range for control histograms
        - xlim_resolution – X-axis range for resolution plots
        - bins – number of histogram bins
        - alpha – histogram transparency
        - layout – plot layout:
            1.stacked
            2.overlay
            3.side by side
        - config_name – configuration directory name
        - mode – processing mode:
            1.raw – direct data
            2.hist – histogram-based processing

## Example usage:
    python run.py config config_0 \
    --xlim_control 100 \
    --xlim_resolution 50 \
    --bins 20 \
    --alpha 0.4 \
    --layout stacked \
    --config_name config_0 \
    --mode raw