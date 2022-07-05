"""Main evaluation function

Loads graphs and evaluates
"""
import logging
import os
import time

import networkx as nx
import daisy

import linajea.tracking
import linajea.utils
from .evaluate import evaluate
from .report import Report

logger = logging.getLogger(__name__)


def evaluate_setup(linajea_config):
    """Evaluates a given setup

    Determines parameters and database to use based on provided configuration.
    Checks if solution has already been computed.
    Loads graphs (ground truth and reconstructed) from databases.
    Calls evaluate on graphs (compute matching and evaluation metrics).
    Writes results to database.
    Returns results.

    Args
    ----
    linajea_config: TrackingConfig
        Tracking configuration object, determines everything it needs
        from config or uses defaults.

    Returns
    -------
    Report
         Report object containing all computed metrics and statistics
    """

    assert len(linajea_config.solve.parameters) == 1, \
        "can only handle single parameter set"
    parameters = linajea_config.solve.parameters[0]

    data = linajea_config.inference_data.data_source
    db_name = data.db_name
    db_host = linajea_config.general.db_host
    logger.debug("ROI used for evaluation: %s %s",
                 linajea_config.evaluate.parameters.roi, data.roi)
    if linajea_config.evaluate.parameters.roi is not None:
        assert linajea_config.evaluate.parameters.roi.shape[0] <= data.roi.shape[0], \
            "your evaluation ROI is larger than your data roi!"
        data.roi = linajea_config.evaluate.parameters.roi
    else:
        linajea_config.evaluate.parameters.roi = data.roi
    evaluate_roi = daisy.Roi(offset=data.roi.offset,
                             shape=data.roi.shape)

    # determine parameters id from database
    results_db = linajea.utils.CandidateDatabase(db_name, db_host)
    parameters_id = results_db.get_parameters_id(parameters,
                                                 fail_if_not_exists=True)

    if not linajea_config.evaluate.from_scratch:
        old_score = results_db.get_score(parameters_id,
                                         linajea_config.evaluate.parameters)
        if old_score:
            logger.info("Already evaluated %d (%s). Skipping",
                        parameters_id, linajea_config.evaluate.parameters)
            score = {}
            for k, v in old_score.items():
                if not isinstance(k, list) or k != "roi":
                    score[k] = v
            logger.debug("Stored results: %s", score)
            report = Report()
            report.__dict__.update(score)
            return report

    logger.info("Evaluating %s in %s",
                os.path.basename(data.datafile.filename)
                if data.datafile is not None else db_name, evaluate_roi)

    edges_db = linajea.utils.CandidateDatabase(db_name, db_host,
                                               parameters_id=parameters_id)

    logger.info("Reading cells and edges in db %s with parameter_id %d",
                db_name, parameters_id)
    start_time = time.time()
    subgraph = edges_db.get_selected_graph(evaluate_roi)

    logger.info("Read %d cells and %d edges in %s seconds",
                subgraph.number_of_nodes(),
                subgraph.number_of_edges(),
                time.time() - start_time)

    if subgraph.number_of_edges() == 0:
        logger.warn("No selected edges for parameters_id %d. Skipping",
                    parameters_id)
        return False

    track_graph = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)

    gt_db = linajea.utils.CandidateDatabase(
        linajea_config.inference_data.data_source.gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s",
                linajea_config.inference_data.data_source.gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db.get_graph(
        evaluate_roi,
    )
    logger.info("Read %d cells and %d edges in %s seconds",
                gt_subgraph.number_of_nodes(),
                gt_subgraph.number_of_edges(),
                time.time() - start_time)

    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %d", parameters_id)
    matching_threshold = linajea_config.evaluate.parameters.matching_threshold
    validation_score = linajea_config.evaluate.parameters.validation_score
    ignore_one_off_div_errors = \
        linajea_config.evaluate.parameters.ignore_one_off_div_errors
    fn_div_count_unconnected_parent = \
        linajea_config.evaluate.parameters.fn_div_count_unconnected_parent
    window_size=linajea_config.evaluate.parameters.window_size

    report = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold,
            linajea_config.general.sparse,
            validation_score,
            window_size,
            ignore_one_off_div_errors,
            fn_div_count_unconnected_parent)

    logger.info("Done evaluating results for %d. Saving results to mongo.",
                parameters_id)
    logger.debug("Result summary: %s", report.get_short_report())
    results_db.write_score(parameters_id, report,
                           eval_params=linajea_config.evaluate.parameters)
    return report
