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

    if args.validate_on_train:
        inference.data_sources = deepcopy(config.train_data.data_sources)

    if args.checkpoint > 0:
        checkpoints = [args.checkpoint]

    if (is_solve or is_evaluate) and args.val_param_id is not None:
        config = fix_val_param_pid(args, config, checkpoints, inference)
    if (is_solve or is_evaluate) and \
       (args.param_id is not None or
        (hasattr(args, "param_ids") and args.param_ids is not None)):
        config = fix_param_pid(args, config, checkpoints, inference)

    max_cell_move = max(config.extract.edge_move_threshold.values())
    for pid in range(len(config.solve.parameters)):
        if config.solve.parameters[pid].max_cell_move is None:
            config.solve.parameters[pid].max_cell_move = max_cell_move

    os.makedirs("tmp_configs", exist_ok=True)
    if hasattr(args, "param_list_idx") and \
       args.param_list_idx is not None:
        param_list_idx = int(args.param_list_idx[1:-1])
        assert param_list_idx <= len(config.solve.parameters), \
            ("invalid index into parameter set list of config, "
             "too large ({}, {})").format(
                 param_list_idx, len(config.solve.parameters))
        solve_parameters = deepcopy(config.solve.parameters[param_list_idx-1])
        config.solve.parameters = [solve_parameters]
    solve_parameters_sets = deepcopy(config.solve.parameters)
    for checkpoint in checkpoints:
        inference_data = {
            'checkpoint': checkpoint,
            'cell_score_threshold': inference.cell_score_threshold}
        for sample in inference.data_sources:
            sample = deepcopy(sample)
            if sample.db_name is None and not config.predict.no_db_access:
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
                if config.solve.write_struct_svm:
                    config.solve.write_struct_svm += "_ckpt_{}_{}".format(
                        checkpoint, os.path.basename(sample.datafile.filename))

            if is_evaluate:
                for solve_parameters in solve_parameters_sets:
                    solve_parameters = deepcopy(solve_parameters)
                    config.solve.parameters = [solve_parameters]
                    config = fix_solve_roi(config)
                    yield config
                continue

            config.path = os.path.join("tmp_configs", "config_{}.toml".format(
                time.time()))
            with open(config.path, 'w') as f:
                toml.dump(attr.asdict(config), f)
            yield config


def fix_val_param_pid(args, config, checkpoints, inference):
    if args.validation:
        sample_name = config.test_data.data_sources[0].datafile.filename
        threshold = config.test_data.cell_score_threshold
    else:
        sample_name = config.validate_data.data_sources[0].datafile.filename
        threshold = config.validate_data.cell_score_threshold
    pid = args.val_param_id

    config = fix_solve_parameters_with_pids(
        config, sample_name, checkpoints[0], inference, [pid],
        threshold=threshold)
    config.solve.parameters[0].val = False
    return config


def fix_param_pid(args, config, checkpoints, inference):
    assert len(checkpoints) == 1, "use param_id to reevaluate a single instance"
    sample_name = inference.data_sources[0].datafile.filename
    if hasattr(args, "param_ids") and args.param_ids is not None:
        pids = list(range(int(args.param_ids[0]), int(args.param_ids[1])+1))
    else:
        pids = [args.param_id]

    config = fix_solve_parameters_with_pids(config, sample_name,
                                            checkpoints[0], inference, pids)
    return config


def fix_solve_roi(config):
    for i in range(len(config.solve.parameters)):
        config.solve.parameters[i].roi = config.inference.data_source.roi
    return config


def fix_solve_parameters_with_pids(config, sample_name, checkpoint, inference,
                                   pids, threshold=None):
    if inference.data_sources[0].db_name is not None:
        db_name = inference.data_sources[0].db_name
    else:
        db_name = checkOrCreateDB(
            config.general.db_host,
            config.general.setup_dir,
            sample_name,
            checkpoint,
            threshold if threshold is not None
            else inference.cell_score_threshold,
            tag=config.general.tag,
            create_if_not_found=False)
    assert db_name is not None, "db for pid {} not found".format(pids)
    pids_t = []
    for pid in pids:
        if isinstance(pid, str):
            if pid[0] in ("\"", "'"):
                pid = int(pid[1:-1])
            else:
                pid = int(pid)
        pids_t.append(pid)
    pids = pids_t

    db = CandidateDatabase(db_name, config.general.db_host)
    config.solve.parameters = []
    parameters_sets = db.get_parameters_many(pids)

    for pid, parameters in zip(pids, parameters_sets):
        if parameters is None:
            continue
        logger.info("getting params %s (id: %s) from database %s (sample: %s)",
                    parameters, pid, db_name, sample_name)
        try:
            solve_parameters = SolveParametersMinimalConfig(**parameters) # type: ignore
            config.solve.non_minimal = False
        except TypeError:
            solve_parameters = SolveParametersNonMinimalConfig(**parameters) # type: ignore
            config.solve.non_minimal = True
        config.solve.parameters.append(solve_parameters)
    return config
