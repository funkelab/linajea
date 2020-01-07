import json
import toml
import os.path


def load_config(f):
    ext = os.path.splitext(f)[1]
    if ext == '.json':
        config = json.load(f)
    elif ext == '.toml':
        config = toml.load(f)
    elif ext == '':
        try:
            config = toml.load(f)
        except ValueError:
            try:
                config = json.load(f)
            except ValueError:
                raise ValueError("No file extension provided "
                                 "and cannot be loaded with json or toml")
    else:
        raise ValueError("Only json and toml config files supported, not %s"
                         % ext)
    return config
