# Documentation of Htt_plotter program:

## General information:
    Program generates Histograms from data it gets when user give configuration(in Configurations folder then changing in run name of the configuration) they want tu use. Core file is plotter which transform data to plots. Data_access is file that changes raw data from files to stream that next is used in plotter. Function in loader.py loading all configurations files, send warrnigs if something is missing and normalize selected configurationes. Script that automatically generates json configurator scans folders in search of data files.
        

## What user can do:
    User specify what data he wants to use and how it should be configurated for his personal desire.

## Specific information about files:

### Plotter.py:
    It contains class plotter which in this file is a main pipline that transform data to plots (control, resolution and comparision of MC and real data). 
    Core methodes are:
        a. load_index - it is building dataset index from configurated samples;
        b. set_parameters - it determines which variables will be ploted;
        c. batch - it creates pairs of variables that will be used in the resolution plots ;
        d. _bin_edges - return edges of bins used in histograms;
        e. _to_numpy - extracts column from batch;
        f. Main pipepline - run_all:
            - reads data;
            - fills histograms;
            - generates plots.

### data_access.py:
    It contains class DataAccess which changes raw data from files (CSV/Parquet) to data stream. Mainly it does: building list of data files, analyze data and shares data as batch stream. 
    Core methodes are:
        a. _resolve_files - changes paths relative to absolute;
        b. _scan_dirs - scans folders for files: CSV or Parquet;
        c. _infer_format - checks if file types are supported in program;
        d. _sample_schema - takes heads of the columns from files;
        e. build_index - creates structure of every dataset;
        f. iter_batches - return data in stream.

### loader.py:
    This  has one function which loads all configuration files requeired by the Plotter, fro given by user configuration. It loads JSON and YAML configurations or if they are missing function raises Error. After loading, the function normalize configurationes to ensure everything exists.

### json_generator.py:
    Script in this file automatically scans MC and Data directiories and creates JSON file that is then used by pipline. 
    It contains functions:
        a. get_color - return a color from the color_palette;
        b. _rel - return a relative path to the project root;
        c. scan_mc_samples - scans MC folders and return samples with file list;
        d. add_data - adds scaned data files to the samples dictionary ;
        e. write_files_json - writes the final JSON configuration file;
        f. main - runs the full pipeline: scanning MC, adding data, and saving the output JSON.

### render.py:
    It saves stacked plots(control, resolution and MC to Data ratio) and defines the visual appearence of plots such as: width, colors, labels, legend, title, grid, stacking order, figure layout.
    It contain functions:
        a. save_stacked_plot - creates and saves stacked histograms(control and resolution plots) and creates their look;
        b. save_data_mc_ratio_plot - saves plot MC to Data ration and creates its look.

### selection.py:
    It is responsible for selection of data and filtration of columns and records based on configuration. It reduces and prepares datasets for plotting and analysis.
    It contain functions:
        a. selection_columns_used - return list of columns required for selection based on configuration;
        b. selection_mask - creates a boolean mask for filtering events using Pandas; 
        c. make_selector - return a function that applies selection to a DataFrame;
        d. make_arrow_filter - builds a PyArrow filter expression for dataset-level filtering;
        e. SELECT - applies selection directly to a DataFrame or file path;
        f. plotting_columns - return available columns split into control and resolution groups for plotting.


### qcd.py:
    It is responsible for estimating QCD background using SS method. It builds QCD templates from data by removing MC from data in SS region, propagates the result to the OS region scaling it by factor.

### Configurations folder:
    Contains Configurations that user want to use for generetaing plots, every Config need to contain 5 files: Config.py, files.json(automatically created by json_generator), variables.json, params.yaml and plotter.yaml.