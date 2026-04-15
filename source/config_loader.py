import json
import yaml
from pathlib import Path


def load_configs(project_root, config_name="config_0"):
    config_dir = project_root / "Configurations" / config_name

    if not config_dir.exists():
        raise FileNotFoundError(f"Config does not exist: {config_dir}")

    with open(config_dir / "files.json", "r", encoding="utf-8") as f:
        sample_config = json.load(f)

    with open(config_dir / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    with open(config_dir / "variables.json", "r", encoding="utf-8") as f:
        variable_config = json.load(f)

    return sample_config, params, variable_config