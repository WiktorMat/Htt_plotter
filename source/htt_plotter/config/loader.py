import json
import logging
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import yaml

logger = logging.getLogger(__name__)

def safe_load_json(path: Path, default):
    if not path.exists():
        logger.warning("Missing file: %s → using default", path)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed loading %s → %s", path, e)
        return default
    
def safe_load_yaml(path: Path, default):
    if not path.exists():
        logger.warning("Missing file: %s → using default", path)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    except Exception as e:
        logger.warning("Failed loading %s → %s", path, e)
        return default

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

    raw = safe_load_json(files_path, default={"samples": {}})

    sample_config = raw.get("samples", raw)

    process_config = safe_load_json(process_path, default={})

    params = safe_load_yaml(params_path, default={})

    if "lumi" not in params:
        logger.warning("Missing 'lumi' in params.yaml → default = 1.0")
        params["lumi"] = 1.0

    variable_config = safe_load_json(variables_path, default={})

    plotter_config = safe_load_yaml(plotter_path, default={})

    # Selection is sourced from Config.py (single source of truth)
    selection_from_config: dict[str, float | int] = {}
    config_py = config_dir / "Config.py"
    if config_py.exists():
        spec = spec_from_file_location(f"htt_plotter.{config_name}.Config", config_py)
        cfg = module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(cfg)  # type: ignore[union-attr]

        attr_to_key = {
            "PT_1_CUT": "pt_1_min",
            "PT_2_CUT": "pt_2_min",
            "ETA_1_CUT": "eta_1_abs_max",
            "ETA_2_CUT": "eta_2_abs_max",
            "IDJET_2_CUT": "idDeepTau2018v2p5VSjet_2_min",
            "IDE_2_CUT": "idDeepTau2018v2p5VSe_2_min",
            "IDMU_2_CUT": "idDeepTau2018v2p5VSmu_2_min",
            "ISO_1_CUT": "iso_1_max",
        }

        for attr, key in attr_to_key.items():
            value = getattr(cfg, attr, None)
            if value is not None:
                selection_from_config[key] = value
    else:
        logger.warning("Config.py not found → using empty selection")
        selection_from_config = {}

    plotter_config["selection"] = selection_from_config
    plotter_config.setdefault("plotting", {})
    plotting = plotter_config["plotting"]

    if not isinstance(plotting, dict):
        logger.warning("plotting section corrupted → resetting")
        plotter_config["plotting"] = {}
        plotting = plotter_config["plotting"]

    plotting.setdefault("control", [])
    plotting.setdefault("resolution", [])

    cleaned_resolution = []
    for item in plotting["resolution"]:
        try:
            cleaned_resolution.append((item[0], item[1]))
        except Exception:
            logger.warning("Invalid resolution entry skipped: %s", item)

    plotting["resolution"] = cleaned_resolution

    return sample_config, params, variable_config, plotter_config, process_config