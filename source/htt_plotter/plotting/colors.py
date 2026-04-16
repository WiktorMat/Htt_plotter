def get_sample_color(sample: str, sample_config: dict, default: str = "tab:blue") -> str:
    cfg = (sample_config or {}).get(sample, {})
    return cfg.get("color", default)
