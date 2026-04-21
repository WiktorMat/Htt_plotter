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

### `plotter.py`
   Contains the `Plotter` class, which is the main pipeline of the project.

    It is responsible for transforming input data into plots.

### Main methods:
    - **load_index()**  
    Builds the dataset index from configured samples.

    - **set_parameters()**  
    Determines which variables will be plotted.

    - **batch()**  
    Creates variable pairs used for resolution plots.

    - **_bin_edges()**  
    Returns histogram bin edges.

    - **_to_numpy()**  
    Extracts selected columns into NumPy arrays.

    - **run_all()**  
    Main execution pipeline:
    - loads data,
    - applies selections,
    - fills histograms,
    - generates and saves plots.

## `data_access.py`
    Contains the `DataAccess` class.

    Responsible for converting raw files (CSV / Parquet) into a streamed batch format used by `Plotter`.

### Main methods:
    - **_resolve_files()**  
    Converts relative paths into absolute paths.

    - **_scan_dirs()**  
    Scans directories for supported files.

    - **_infer_format()**  
    Detects file type.

    - **_sample_schema()**  
    Reads column names and schema.

    - **build_index()**  
    Creates dataset metadata.

    - **iter_batches()**  
    Returns data in streaming batches.

## `loader.py`
    Loads all configuration files required by the program.

    Supported formats:
    - JSON
    - YAML
    - Python config files

    If required files are missing, an error is raised.

    The loader also validates and normalizes configuration content.

## `json_generator.py`
    Automatically scans data directories and creates `files.json`.

    This file defines available datasets and their file locations.

### Main functions:
    - **get_color()** – assigns colors,
    - **_rel()** – creates relative paths,
    - **scan_mc_samples()** – scans MC samples,
    - **add_data()** – adds real data,
    - **write_files_json()** – saves configuration,
    - **main()** – runs full generation pipeline.

## `render.py`
    Responsible for saving plots and defining visual appearance.

    Controls:
    - colors,
    - legends,
    - titles,
    - labels,
    - grids,
    - layout,
    - stacking order.

### Main functions:
    - **save_stacked_plot()**
    - **save_data_mc_ratio_plot()**

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

# How to run the program using test_run.py
    To run the program with custom parameters, configure the following seven options: xlim_control, xlim_resolution, bins, alpha, layout and config_name, mode.
        - xlim_control - sets the X-axis limit for control histograms,
        - xlim_resolution - sets the X-axis limit for resolution histograms,
        - bins is setting - defines the number of bins used in the histograms,
        - alpha - controls the transparency of histogram bars,
        - layout - determines how multiple histograms are displayed, available values: stacked, overlay, side by side,
        - config_name - specifies which configuration directory should be used,
        - mode - selects how data is processed and plotted, available modes: raw or hist