from copy import deepcopy
import logging
import os
import time

import attr
import toml

from linajea import (CandidateDatabase,
                     checkOrCreateDB)
from linajea.config import (InferenceDataTrackingConfig,
                            SolveParametersMinimalConfig,
                            SolveParametersNonMinimalConfig,
                            TrackingConfig,
                            maybe_fix_config_paths_to_machine_and_load)

logger = logging.getLogger(__name__)


def getNextInferenceData(args, is_solve=False, is_evaluate=False):
    config = maybe_fix_config_paths_to_machine_and_load(args.config)
    config = TrackingConfig(**config)

    if args.validation:
        inference = deepcopy(config.validate_data)
        checkpoints = config.validate_data.checkpoints
    else:
        inference = deepcopy(config.test_data)
        checkpoints = [config.test_data.checkpoint]

    if args.checkpoint > 0:
        checkpoints = [args.checkpoint]

    if is_solve and args.val_param_id is not None:
        config = fix_solve_pid(args, config, checkpoints, inference)
    if is_evaluate and args.param_id is not None:
        config = fix_evaluate_pid(args, config, checkpoints, inference)

    max_cell_move = max(config.extract.edge_move_threshold.values())
    for pid in range(len(config.solve.parameters)):
        if config.solve.parameters[pid].max_cell_move is None:
            config.solve.parameters[pid].max_cell_move = max_cell_move

    os.makedirs("tmp_configs", exist_ok=True)
    solve_parameters_sets = deepcopy(config.solve.parameters)
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
                    inference.cell_score_threshold,
                    tag=config.general.tag)
            inference_data['data_source'] = sample
            config.inference = InferenceDataTrackingConfig(**inference_data) # type: ignore
            if is_solve:
                config = fix_solve_roi(config)

            if is_evaluate:
                config = fix_evaluate_roi(config)
                for solve_parameters in solve_parameters_sets:
                    solve_parameters = deepcopy(solve_parameters)
                    solve_parameters.roi = config.inference.data_source.roi
                    config.solve.parameters = [solve_parameters]
                    yield config
                continue

            config.path = os.path.join("tmp_configs", "config_{}.toml".format(
                time.time()))
            with open(config.path, 'w') as f:
                toml.dump(attr.asdict(config), f)
            yield config


def fix_solve_pid(args, config, checkpoints, inference):
    if args.param_id is not None:
        assert len(checkpoints) == 1, "use param_id to reevaluate a single instance"
        sample_name = inference.data_sources[0].datafile.filename,
        pid = args.param_id
    else:
        assert not args.validation, "use val_param_id to apply validation parameters with that ID to test set not validation set"
        sample_name = config.validate_data.data_sources[0].datafile.filename
        pid = args.val_param_id

    config = fix_solve_parameters_with_pid(config, sample_name, checkpoints[0],
                                           inference, pid)

    return config


def fix_solve_roi(config):
    for i in range(len(config.solve.parameters)):
        config.solve.parameters[i].roi = config.inference.data_source.roi
    return config


def fix_evaluate_pid(args, config, checkpoints, inference):
    assert len(checkpoints) == 1, "use param_id to reevaluate a single instance"
    sample_name = inference.data_sources[0].datafile.filename
    pid = args.param_id

    config = fix_solve_parameters_with_pid(config, sample_name, checkpoints[0],
                                           inference, pid)
    return config

def fix_evaluate_roi(config):
    if config.evaluate.parameters.roi is not None:
        assert config.evaluate.parameters.roi.shape[0] < \
            config.inference.data_source.roi.shape[0], \
            "your evaluation ROI is larger than your data roi!"
        config.inference.data_source.roi = config.evaluate.parameters.roi
    return config



def fix_solve_parameters_with_pid(config, sample_name, checkpoint, inference,
                                  pid):
    db_name = checkOrCreateDB(
        config.general.db_host,
        config.general.setup_dir,
        sample_name,
        checkpoint,
        inference.cell_score_threshold)
    db = CandidateDatabase(db_name, config.general.db_host)
    parameters = db.get_parameters(pid)
    logger.info("getting params %s (id: %s) from database %s (sample: %s)",
                parameters, pid, db_name, sample_name)
    try:
        solve_parameters = [SolveParametersMinimalConfig(**parameters)] # type: ignore
        config.solve.non_minimal = False
    except TypeError:
        solve_parameters = [SolveParametersNonMinimalConfig(**parameters)] # type: ignore
        config.solve.non_minimal = True
    config.solve.parameters = solve_parameters
    return config
