"""Provides facility to automatically loop over all desired samples

Useful if data set (validate_data or test_data) contains multiple samples,
or if multiple checkpoints should be compared or in general to be able to
set (monolithic) config file once and compute all steps without changing it
or needing different ones for validation and testing.
"""
from copy import deepcopy
import logging
import os
import time

import attr
import toml

from linajea.utils import (CandidateDatabase,
                           checkOrCreateDB)
from linajea.config import (InferenceDataTrackingConfig,
                            SolveParametersConfig,
                            TrackingConfig,
                            maybe_fix_config_paths_to_machine_and_load)

logger = logging.getLogger(__name__)


def getNextInferenceData(args, is_solve=False, is_evaluate=False):
    """

    Args
    ----
    args: argparse.Namespace or types.SimpleNamespace
        Simple Namespace object with some (mostly optional, depending on
        step that should be computed) attributes, e.g., the result of
        calling parse_args on an argparse.ArgumentParser or by constructing
        a types.SimpleNamespace manually.

        Attributes
        ----------
        config: Path
            Mandatory,path to the configuration file that should be used
        validation: bool
            Compute results on validate_data (or on test_data)
        validate_on_train: bool
            Use train_data for validation, for debugging/checking for
            overfitting
        checkpoint: int
            Can be used to overwrite value of checkpoint contained in config
        param_list_idx: int
            Index into list of parameter sets in config.solve.parameters
            Only this element will be computed.
        val_param_id: int
            Load parameters from validation database with this ID, use it for
            test data; only works with a single data sample (otherwise ID might
            not be unique)
        param_id: int
            Load parameters from current database with this ID and compute
        param_ids: list of int
            Load sets of parameters with these IDs from current database
            and compute; if two elements, interpreted as range; if more
            than two elements, interpreted as list.
    is_solve: bool
        Compute solving step
    is_evaluate: bool
        Compute evaluation step
    """
    config = maybe_fix_config_paths_to_machine_and_load(args.config)
    config = TrackingConfig(**config)

    if hasattr(args, "validation") and args.validation:
        inference_data = deepcopy(config.validate_data)
        checkpoints = config.validate_data.checkpoints
    else:
        inference_data = deepcopy(config.test_data)
        checkpoints = [config.test_data.checkpoint]

    if hasattr(args, "validate_on_train") and args.validate_on_train:
        inference_data.data_sources = deepcopy(config.train_data.data_sources)

    if hasattr(args, "checkpoint") and args.checkpoint > 0:
        checkpoints = [args.checkpoint]

    if config.solve.parameters is not None:
        max_cell_move = (max(config.extract.edge_move_threshold.values())
                         if config.extract is not None else None)
        for pid in range(len(config.solve.parameters)):
            if config.solve.parameters[pid].max_cell_move is None:
                assert max_cell_move is not None, (
                    "Please provide a max_cell_move value, either as "
                    "extract.edge_move_threshold or directly in the parameter "
                    "sets in solve.parameters! (What is the maximum distance  "
                    "that a cell can move between two adjacent frames?)")
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
        if hasattr(args, "val_param_id") and (is_solve or is_evaluate) and \
           args.val_param_id is not None:
            config = _fix_val_param_pid(args, config, checkpoint)
            solve_parameters_sets = deepcopy(config.solve.parameters)
        if ((is_solve or is_evaluate) and
            ((hasattr(args, "param_id") and (args.param_id is not None)) or
             (hasattr(args, "param_ids") and args.param_ids is not None))):
            config = _fix_param_pid(args, config, checkpoint, inference_data)
            solve_parameters_sets = deepcopy(config.solve.parameters)
        inference_data_tmp = {
            'checkpoint': checkpoint,
            'cell_score_threshold': inference_data.cell_score_threshold}
        for sample in inference_data.data_sources:
            sample = deepcopy(sample)
            if sample.db_name is None and hasattr(config, "predict") and \
               not config.predict.no_db_access:
                sample.db_name = checkOrCreateDB(
                    config.general.db_host,
                    config.general.setup_dir,
                    sample.datafile.filename,
                    checkpoint,
                    inference_data.cell_score_threshold,
                    roi=attr.asdict(sample.roi),
                    tag=config.general.tag)
            inference_data_tmp['data_source'] = sample
            config.inference_data = InferenceDataTrackingConfig(
                **inference_data_tmp)  # type: ignore
            if is_solve:
                config = _fix_solve_roi(config)

            if is_evaluate:
                for solve_parameters in solve_parameters_sets:
                    solve_parameters = deepcopy(solve_parameters)
                    config.solve.parameters = [solve_parameters]
                    config = _fix_solve_roi(config)
                    yield config
                continue

            config.path = os.path.join("tmp_configs", "config_{}.toml".format(
                time.time()))
            with open(config.path, 'w') as f:
                toml.dump(attr.asdict(config), f)
            yield config


def _fix_val_param_pid(args, config, checkpoint):
    if hasattr(args, "validation") and args.validation:
        tmp_data = config.test_data
    else:
        tmp_data = config.validate_data
    assert len(tmp_data.data_sources) == 1, (
        "val_param_id only supported with a single sample")
    if tmp_data.data_sources[0].db_name is None:
        db_meta_info = {
            "sample": tmp_data.data_sources[0].datafile.filename,
            "iteration": checkpoint,
            "cell_score_threshold": tmp_data.cell_score_threshold,
            "roi": tmp_data.data_sources[0].roi
        }
        db_name = None
    else:
        db_name = tmp_data.data_sources[0].db_name
        db_meta_info = None

    pid = args.val_param_id

    config = _fix_solve_parameters_with_pids(
        config, [pid], db_meta_info, db_name)
    config.solve.parameters[0].val = False
    return config


def _fix_param_pid(args, config, checkpoint, inference_data):
    assert len(inference_data.data_sources) == 1, (
        "param_id(s) only supported with a single sample")
    if inference_data.data_sources[0].db_name is None:
        db_meta_info = {
            "sample": inference_data.data_sources[0].datafile.filename,
            "iteration": checkpoint,
            "cell_score_threshold": inference_data.cell_score_threshold,
            "roi": inference_data.data_sources[0].roi
        }
        db_name = None
    else:
        db_name = inference_data.data_sources[0].db_name
        db_meta_info = None

    if hasattr(args, "param_ids") and args.param_ids is not None:
        if len(args.param_ids) == 2:
            pids = list(range(int(args.param_ids[0]),
                              int(args.param_ids[1])+1))
        else:
            pids = args.param_ids
    else:
        pids = [args.param_id]

    config = _fix_solve_parameters_with_pids(config, pids, db_meta_info,
                                             db_name)
    return config


def _fix_solve_roi(config):
    for i in range(len(config.solve.parameters)):
        config.solve.parameters[i].roi = config.inference_data.data_source.roi
    return config


def _fix_solve_parameters_with_pids(config, pids, db_meta_info=None,
                                    db_name=None):
    if db_name is None:
        db_name = checkOrCreateDB(
            config.general.db_host,
            config.general.setup_dir,
            db_meta_info["sample"],
            db_meta_info["iteration"],
            db_meta_info["cell_score_threshold"],
            roi=attr.asdict(db_meta_info["roi"]),
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
                    parameters, pid, db_name,
                    db_meta_info["sample"] if db_meta_info is not None
                    else None)
        solve_parameters = SolveParametersConfig(**parameters)  # type: ignore
        config.solve.parameters.append(solve_parameters)
    return config
