import json
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_configs(project_root: Path, config_name: str = "config_0"):
    project_root = Path(project_root)
    config_dir = project_root / "Configurations" / config_name
    source_dir = project_root / "source"

    files_path = config_dir / "files.json"
    params_path = config_dir / "params.yaml"
    variables_path = config_dir / "variables.json"
    plotter_path = config_dir / "plotter.yaml"
    process_path = config_dir / "process.json"

    if not files_path.exists():
        files_path = source_dir / "files.json"
    if not params_path.exists():
        params_path = source_dir / "params.yaml"
    if not variables_path.exists():
        variables_path = source_dir / "variables.json"
    if not process_path.exists():
        process_path = source_dir / "process.json"

    logger.info("[CONFIG]: %s", config_name)
    logger.info("[FILES]: %s", files_path)
    logger.info("[PARAMS]: %s", params_path)
    logger.info("[VARIABLES]: %s", variables_path)
    logger.info("[PLOTTER]: %s", plotter_path)
    logger.info("[PROCESS]: %s", process_path)

    with open(files_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sample_config = raw.get("samples", raw)

    with open(process_path, "r", encoding="utf-8") as f:
        process_config = json.load(f)

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    with open(variables_path, "r", encoding="utf-8") as f:
        variable_config = json.load(f)

    with open(plotter_path, "r", encoding="utf-8") as f:
        plotter_config = yaml.safe_load(f) or {}

    plotter_config.setdefault("selection", {})
    plotter_config.setdefault("plotting", {})
    plotting = plotter_config["plotting"]

    plotting.setdefault("control", [])
    plotting.setdefault("resolution", [])

    cleaned_resolution = []
    for item in plotting["resolution"]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            cleaned_resolution.append(tuple(item))
        else:
            logger.warning(
                "Invalid resolution entry skipped: %s (expected [c, r])",
                item,
            )

    plotting["resolution"] = cleaned_resolution

    return sample_config, params, variable_config, plotter_config, process_config