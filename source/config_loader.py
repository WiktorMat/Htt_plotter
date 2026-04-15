import json
import yaml
from pathlib import Path

def load_configs(project_root):
    config_dir = project_root / "Configurations" / "config_0"

    with open(config_dir / "files.json", "r") as f:
        sample_config = json.load(f)

    with open(config_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    with open(config_dir / "variables.json", "r") as f:
        variable_config = json.load(f)

    return sample_config, params, variable_config