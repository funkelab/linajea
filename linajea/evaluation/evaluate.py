from .match import match_edges
from .evaluator import Evaluator
import logging

logger = logging.getLogger(__name__)


def evaluate(
        gt_track_graph,
        rec_track_graph,
        matching_threshold,
        **kwargs):
    logger.info("Matching GT edges to REC edges...")
    gt_edges, rec_edges, edge_matches, unselected_potential_matches = match_edges(
            gt_track_graph,
            rec_track_graph,
            matching_threshold)
    logger.info("Done matching")
    edge_matches = [(gt_edges[gt_ind], rec_edges[rec_ind])
                    for gt_ind, rec_ind in edge_matches]
    evaluator = Evaluator(
            gt_track_graph,
            rec_track_graph,
            edge_matches,
            unselected_potential_matches)
    evaluator.evaluate(**kwargs)
    return evaluator
