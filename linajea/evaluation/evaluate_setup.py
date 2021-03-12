import linajea.tracking
from .evaluate import evaluate
from ..datasets import get_source_roi
import logging
import daisy
import time
import sys

logger = logging.getLogger(__name__)


def evaluate_setup(
        sample,
        db_host,
        db_name,
        gt_db_name,
        matching_threshold=None,
        frames=None,
        limit_to_roi=None,
        from_scratch=True,
        sparse=True,
        data_dir='../01_data',
        **kwargs):

    parameters = linajea.tracking.TrackingParameters(**kwargs)
    if matching_threshold is None:
        logger.error("No matching threshold for evaluation")
        sys.exit()

    # determine parameters id from database
    results_db = linajea.CandidateDatabase(
        db_name,
        db_host)
    parameters_id = results_db.get_parameters_id(parameters)

    if not from_scratch:
        old_score = results_db.get_score(parameters_id, frames=frames)
        if old_score:
            logger.info("Already evaluated %d (frames: %s). Skipping" %
                        (parameters_id, frames))
            return old_score

    voxel_size, source_roi = get_source_roi(data_dir, sample)

    # limit to specific frames, if given
    if frames:
        begin, end = frames
        crop_roi = daisy.Roi(
            (begin, None, None, None),
            (end - begin, None, None, None))
        source_roi = source_roi.intersect(crop_roi)

    # limit to roi, if given
    if limit_to_roi:
        source_roi.intersect(limit_to_roi)

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
    report = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=matching_threshold,
            sparse=sparse)

    logger.info("Done evaluating results for %d. Saving results to mongo."
                % parameters_id)
    results_db.write_score(parameters_id, report, frames=frames)
