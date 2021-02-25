import json
import toml
import os.path
import logging
from linajea.tracking import TrackingParameters

logger = logging.getLogger(__name__)


def load_config(config_file):
    ext = os.path.splitext(config_file)[1]
    with open(config_file, 'r') as f:
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
            raise ValueError("Only json and toml config files supported,"
                             " not %s" % ext)
    return config


def tracking_params_from_config(config):
    solve_config = {}
    solve_config.update(config['general'])
    solve_config.update(config['solve'])
    if 'version' not in solve_config:
        version = solve_config['singularity_image'].split(':')[-1]
        solve_config['version'] = version
    logger.debug("Version: %s" % solve_config['version'])
    solve_config.update({
        'max_cell_move': config['extract_edges']['edge_move_threshold']})

    return TrackingParameters(**solve_config)
