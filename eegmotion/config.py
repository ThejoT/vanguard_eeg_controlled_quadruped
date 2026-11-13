from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    win = cfg["online"]["window"]
    span = cfg["epoch"]["tmax"] - cfg["epoch"]["tmin"]
    if abs(win - span) > 1e-6:
        raise ValueError(f"online.window ({win}) must equal epoch.tmax - epoch.tmin ({span}) "
                         "so the live decoder sees windows shaped like the training data")
    return cfg
