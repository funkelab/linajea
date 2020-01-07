import json
import toml


def load_config(f):
    if f.endswith('.json'):
        config = json.load(f)
    elif f.endswith('.toml'):
        config = toml.load(f)
    else:
        raise ValueError("Only json and toml config files supported")
    return config
