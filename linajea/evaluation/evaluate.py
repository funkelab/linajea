from .match import match_edges
from .evaluator import Evaluator
import logging

logger = logging.getLogger(__name__)


def evaluate(
        gt_track_graph,
        rec_track_graph,
        matching_threshold,
        sparse,
        validation_score=False,
        window_size=50,
        ignore_one_off_div_errors=False,
        fn_div_count_unconnected_parent=True):
    ''' Performs both matching and evaluation on the given
    gt and reconstructed tracks, and returns a Report
    with the results.
    '''
    logger.info("Checking validity of reconstruction")
    Evaluator.check_track_validity(rec_track_graph)
    logger.info("Matching GT edges to REC edges...")
    gt_edges, rec_edges, edge_matches, unselected_potential_matches =\
        match_edges(
            gt_track_graph,
            rec_track_graph,
            matching_threshold)
    logger.info("Done matching. Evaluating")
    edge_matches = [(gt_edges[gt_ind], rec_edges[rec_ind])
                    for gt_ind, rec_ind in edge_matches]
    evaluator = Evaluator(
            gt_track_graph,
            rec_track_graph,
            edge_matches,
            unselected_potential_matches,
            sparse=sparse,
            validation_score=validation_score,
            window_size=window_size,
            ignore_one_off_div_errors=ignore_one_off_div_errors,
            fn_div_count_unconnected_parent=fn_div_count_unconnected_parent)
    return evaluator.evaluate()
