from copy import deepcopy
import os
import time

import attr
import toml

from linajea import checkOrCreateDB
from linajea.config import (InferenceDataTrackingConfig,
                            TrackingConfig)

# TODO: better name maybe?
def getNextInferenceData(args):
    config = TrackingConfig.from_file(args.config)

    if args.validation:
        inference = deepcopy(config.validate_data)
        checkpoints = config.validate_data.checkpoints
    else:
        inference = deepcopy(config.test_data)
        checkpoints = [config.test_data.checkpoint]

    if args.checkpoint > 0:
        checkpoints = [args.checkpoint]

    os.makedirs("tmp_configs", exist_ok=True)
    for checkpoint in checkpoints:
        inference_data = {
            'checkpoint': checkpoint,
            'cell_score_threshold': inference.cell_score_threshold}
        for sample in inference.data_sources:
            if sample.db_name is None:
                sample.db_name = checkOrCreateDB(
                    config.general.db_host,
                    config.general.setup_dir,
                    sample.datafile.filename,
                    checkpoint,
                    inference.cell_score_threshold)
            inference_data['data_source'] = sample
            config.inference = InferenceDataTrackingConfig(**inference_data) # type: ignore
            config.path = os.path.join("tmp_configs", "config_{}.toml".format(time.time()))
            with open(config.path, 'w') as f:
                toml.dump(attr.asdict(config), f)
            yield config
