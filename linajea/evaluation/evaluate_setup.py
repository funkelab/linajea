import linajea.tracking
from .evaluate import evaluate
import logging
import daisy
import time
import os
import json

logger = logging.getLogger(__name__)


def evaluate_setup(
        sample,
        db_host,
        db_name,
        gt_db_name,
        matching_threshold,
        frames=None,
        from_scratch=True,
        error_details=False,
        **kwargs):

    parameters = linajea.tracking.TrackingParameters(**kwargs)

    # determine parameters id from database
    results_db = linajea.CandidateDatabase(
        db_name,
        db_host)
    parameters_id = results_db.get_parameters_id(parameters)

    if not from_scratch:
        old_score = results_db.get_score(parameters_id)
        if old_score:
            logger.info("Already evaluated %d. Skipping" % parameters_id)
            return old_score

    data_dir = '../01_data'

    # get absolute paths
    if os.path.isfile(sample) or sample.endswith((".zarr", ".n5")):
        sample_dir = os.path.abspath(os.path.join(data_dir,
                                                  os.path.dirname(sample)))
    else:
        sample_dir = os.path.abspath(os.path.join(data_dir, sample))

    # get ROI of source
    with open(os.path.join(sample_dir, 'attributes.json'), 'r') as f:
        attributes = json.load(f)

    voxel_size = daisy.Coordinate(attributes['resolution'])
    shape = daisy.Coordinate(attributes['shape'])
    offset = daisy.Coordinate(attributes['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)

    # limit to specific frames, if given
    if frames:
        begin, end = frames
        crop_roi = daisy.Roi(
            (begin, None, None, None),
            (end - begin, None, None, None))
        source_roi = source_roi.intersect(crop_roi)

    logger.info("Evaluating in %s", source_roi)

    edges_db = linajea.CandidateDatabase(
            db_name, db_host, parameters_id=parameters_id)

    logger.info("Reading cells and edges in db %s with parameter_id %d"
                % (db_name, parameters_id))
    start_time = time.time()
    subgraph = edges_db.get_selected_graph(source_roi)

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

    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s"
                % gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db[source_roi]
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))
    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %d" % parameters_id)
    evaluator = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=matching_threshold,
            error_details=error_details)

    logger.info("Done evaluating results for %d. Saving results to mongo."
                % parameters_id)
    results_db.write_score(parameters_id, evaluator)
    logger.info(evaluator)
