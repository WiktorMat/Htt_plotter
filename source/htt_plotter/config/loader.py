import json
from pathlib import Path

import yaml


def load_configs(project_root: Path, config_name: str = "config_0"):
    """Load plotter configuration for a given config name.

    Returns:
        sample_config: dict from files.json
        params: dict from params.yaml
        variable_config: dict from variables.json
        plotter_config: dict from plotter.yaml (selection + plotting groups)
    """

    project_root = Path(project_root)
    config_dir = project_root / "Configurations" / config_name
    source_dir = project_root / "source"

    # Allow reading files.json/params.yaml/variables.json from source/ as fallback.
    files_path = config_dir / "files.json"
    if not files_path.exists():
        files_path = source_dir / "files.json"

    params_path = config_dir / "params.yaml"
    if not params_path.exists():
        params_path = source_dir / "params.yaml"

    variables_path = config_dir / "variables.json"
    if not variables_path.exists():
        variables_path = source_dir / "variables.json"

    plotter_path = config_dir / "plotter.yaml"

    if not files_path.exists():
        raise FileNotFoundError(
            f"Missing files.json (looked in {config_dir} and {source_dir})"
        )
    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing params.yaml (looked in {config_dir} and {source_dir})"
        )
    if not variables_path.exists():
        raise FileNotFoundError(
            f"Missing variables.json (looked in {config_dir} and {source_dir})"
        )
    if not plotter_path.exists():
        raise FileNotFoundError(
            f"Missing plotter.yaml (looked in {config_dir}). "
            f"Create it to define selection and plotting groups for '{config_name}'."
        )

    with open(files_path, "r", encoding="utf-8") as f:
        sample_config = json.load(f)

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    with open(variables_path, "r", encoding="utf-8") as f:
        variable_config = json.load(f)

    with open(plotter_path, "r", encoding="utf-8") as f:
        plotter_config = yaml.safe_load(f) or {}

    # Normalize minimal shape.
    plotter_config.setdefault("selection", {})
    plotter_config.setdefault("plotting", {})
    plotter_config["plotting"].setdefault("control", [])
    plotter_config["plotting"].setdefault("resolution", [])

    return sample_config, params, variable_config, plotter_config
