from __future__ import absolute_import
from copy import deepcopy
import json
import logging
import os
import sys
import time

import daisy
import linajea.tracking
from linajea import (adjust_postprocess_roi,
                     checkOrCreateDB,
                     load_config)
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def get_eval_params_dict(**kwargs):
    assert 'matching_threshold' in kwargs['evaluation'], \
        "No matching threshold for evaluation, check config"
    eval_params = {
        'matching_threshold': kwargs['evaluation']['matching_threshold']}
    if 'frames' in kwargs['postprocessing']:
        eval_params['frame_start'] = kwargs['postprocessing']['frames'][0]
        eval_params['frame_end'] = kwargs['postprocessing']['frames'][1]
    else:
        eval_params['frame_start'] = False
        eval_params['frame_end'] = False

    if 'limit_to_roi_offset' in kwargs['postprocessing']:
        assert 'limit_to_roi_shape' in kwargs['postprocessing'],\
            "Must specify shape and offset in config file"
        eval_params['limit_to_roi_offset'] = \
            kwargs['postprocessing']['limit_to_roi_offset']
        eval_params['limit_to_roi_shape'] = \
            kwargs['postprocessing']['limit_to_roi_shape']
    else:
        eval_params['limit_to_roi_offset'] = False
        eval_params['limit_to_roi_shape'] = False

    eval_params['sparse'] = kwargs['evaluation']['sparse']
    return eval_params


def evaluate_setup(**kwargs):
    if kwargs.get('validation'):
        samples = kwargs['data']['val_data_dirs']
    else:
        samples = kwargs['data']['test_data_dirs']

    for sample in samples:
        sample_kwargs = deepcopy(kwargs)
        evaluate_setup_sample(sample=sample, **sample_kwargs)

def evaluate_setup_sample(**kwargs):
    if 'db_name' not in kwargs['general']:
        kwargs['general']['db_name'] = checkOrCreateDB(**kwargs)

    eval_params = get_eval_params_dict(**kwargs)

    track_params = linajea.tracking.TrackingParameters(**kwargs['solve'])
    # determine parameters id from database
    results_db = linajea.CandidateDatabase(
        kwargs['general']['db_name'],
        kwargs['general']['db_host'])
    parameters_id = results_db.get_parameters_id(track_params)

    logger.info("from scratch %s", kwargs['evaluation']['from_scratch'])
    if not kwargs['evaluation']['from_scratch']:
        old_score = results_db.get_score(parameters_id,
                                         eval_params=eval_params)
        if old_score:
            logger.info("Already evaluated %d (eval_params: %s). Skipping",
                        parameters_id, eval_params)
            return old_score

    # get ROI of source
    data_config = load_config(os.path.join(kwargs['sample'],
                                           "data_config.toml"))
    voxel_size = daisy.Coordinate(kwargs['data']['voxel_size'])
    shape = daisy.Coordinate(data_config['general']['shape'])
    offset = daisy.Coordinate(data_config['general']['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)

    # limit to specific frames/roi, if given
    source_roi = adjust_postprocess_roi(source_roi, **kwargs)
    logger.info("Limiting evaluation to roi %s", source_roi)

    edges_db = linajea.CandidateDatabase(
        kwargs['general']['db_name'],
        kwargs['general']['db_host'],
        parameters_id=parameters_id)

    logger.info("Reading cells and edges in db %s with parameter_id %d",
                kwargs['general']['db_name'], parameters_id)
    start_time = time.time()
    subgraph = edges_db.get_selected_graph(source_roi)

    logger.info("Read %d cells and %d edges in %s seconds",
                subgraph.number_of_nodes(),
                subgraph.number_of_edges(),
                time.time() - start_time)

    if subgraph.number_of_edges() == 0:
        logger.warn("No selected edges for parameters_id %d. Skipping",
                    parameters_id)
        return
    track_graph = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)

    gt_db = linajea.CandidateDatabase(kwargs['evaluation']['gt_db_name'], kwargs['general']['db_host'])

    logger.info("Reading ground truth cells and edges in db %s",
                kwargs['evaluation']['gt_db_name'])
    start_time = time.time()
    gt_subgraph = gt_db[source_roi]
    logger.info("Read %d cells and %d edges in %s seconds",
                gt_subgraph.number_of_nodes(),
                gt_subgraph.number_of_edges(),
                time.time() - start_time)
    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %d", parameters_id)
    report = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=kwargs['evaluation']['matching_threshold'],
            sparse=kwargs['evaluation']['sparse'])

    logger.info("Done evaluating results for %d. Saving results to mongo.",
                parameters_id)
    results_db.write_score(parameters_id, report, eval_params=eval_params)
