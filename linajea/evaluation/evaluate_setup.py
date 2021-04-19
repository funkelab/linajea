import logging
import os
import sys
import time

import daisy

import linajea.tracking
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def evaluate_setup(linajea_config):

    assert len(linajea_config.solve.parameters) == 1, \
        "can only handle single parameter set"
    parameters = linajea_config.solve.parameters[0]

    data = linajea_config.inference.data_source
    db_name = data.db_name
    db_host = linajea_config.general.db_host
    evaluate_roi = daisy.Roi(offset=data.roi.offset,
                             shape=data.roi.shape)

    # determine parameters id from database
    results_db = linajea.CandidateDatabase(db_name, db_host)
    parameters_id = results_db.get_parameters_id(parameters)

    if not linajea_config.evaluate.from_scratch:
        old_score = results_db.get_score(parameters_id,
                                         linajea_config.evaluate.parameters)
        if old_score:
            logger.info("Already evaluated %d (%s). Skipping" %
                        (parameters_id, linajea_config.evaluate.parameters))
            logger.info("Stored results: %s", old_score)
            return

    logger.info("Evaluating %s in %s",
                os.path.basename(data.datafile.filename), evaluate_roi)

    edges_db = linajea.CandidateDatabase(db_name, db_host,
                                         parameters_id=parameters_id)

    logger.info("Reading cells and edges in db %s with parameter_id %d"
                % (db_name, parameters_id))
    start_time = time.time()
    subgraph = edges_db.get_selected_graph(evaluate_roi)

    logger.info("Read %d cells and %d edges in %s seconds"
                % (subgraph.number_of_nodes(),
                   subgraph.number_of_edges(),
                   time.time() - start_time))

    if subgraph.number_of_edges() == 0:
        logger.warn("No selected edges for parameters_id %d. Skipping"
                    % parameters_id)
        return
    track_graph = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)

    gt_db = linajea.CandidateDatabase(
        linajea_config.inference.data_source.gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s"
                % linajea_config.inference.data_source.gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db[evaluate_roi]
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))
    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %d" % parameters_id)
    report = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=linajea_config.evaluate.parameters.matching_threshold,
            sparse=linajea_config.general.sparse)

    logger.info("Done evaluating results for %d. Saving results to mongo."
                % parameters_id)
    logger.info("Result summary: %s", report.get_short_report())
    results_db.write_score(parameters_id, report,
                           eval_params=linajea_config.evaluate.parameters)
