from .match import match_edges
from .evaluator import Evaluator
import logging

logger = logging.getLogger(__name__)


def evaluate(
        gt_track_graph,
        rec_track_graph,
        matching_threshold,
        error_details=False,
        sparse=True,
        calc_aeftl=False):

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
    evaluator.get_fn_edges()
    evaluator.get_fp_edges(sparse=sparse)
    evaluator.get_identity_switches()
    evaluator.get_fp_divisions(sparse=sparse)
    evaluator.get_fn_divisions(count_fn_edges=True)
    evaluator.get_f_score()
    if calc_aeftl:
        evaluator.get_aeftl_and_erl()
    return evaluator
