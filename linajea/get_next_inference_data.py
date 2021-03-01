from copy import deepcopy
import os
import time

import attr
import toml

from linajea import (CandidateDatabase,
                     checkOrCreateDB)
from linajea.config import (InferenceDataTrackingConfig,
                            SolveParametersConfig,
                            TrackingConfig)

# TODO: better name maybe?
def getNextInferenceData(args, val_param_id=None):
    config = TrackingConfig.from_file(args.config)

    if args.validation:
        inference = deepcopy(config.validate_data)
        checkpoints = config.validate_data.checkpoints
    else:
        inference = deepcopy(config.test_data)
        checkpoints = [config.test_data.checkpoint]

    if args.checkpoint > 0:
        checkpoints = [args.checkpoint]

    if val_param_id is not None:
        assert not args.validation, "use val_param_id to apply validation parameters with that ID to test set not validation set"

        val_db_name = checkOrCreateDB(
            config.general.db_host,
            config.general.setup_dir,
            config.validate_data.data_sources[0].datafile.filename,
            checkpoints[0],
            inference.cell_score_threshold)
        val_db_name = checkOrCreateDB(config, val_sample)
        val_db = CandidateDatabase(
            val_db_name, config['general']['db_host'])
        parameters = val_db.get_parameters(val_param_id)
        logger.info("getting params %s (id: %s) from validation database %s (sample: %s)",
                    parameters, val_param_id, val_db_name,
                    config.validate_data.data_sources[0].datafile.filename)
        solve_parameters = [SolveParametersConfig(**parameters)]
        config.solve.parameters = solve_parameters

    max_cell_move = max(config.extract.edge_move_threshold.values())
    for param_id in range(len(config.solve.parameters)):
        if config.solve.parameters[param_id].max_cell_move is None:
            config.solve.parameters[param_id].max_cell_move = max_cell_move

    os.makedirs("tmp_configs", exist_ok=True)
    for checkpoint in checkpoints:
        inference_data = {
            'checkpoint': checkpoint,
            'cell_score_threshold': inference.cell_score_threshold}
        for sample in inference.data_sources:
            sample = deepcopy(sample)
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
