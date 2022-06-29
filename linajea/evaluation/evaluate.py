"""Compares two graphs and evaluates quality of matching

Typical use case:
Match ground truth graph to reconstructed graph and evaluate result
"""
import logging

from .match import match_edges
from .evaluator import Evaluator


logger = logging.getLogger(__name__)


def evaluate(
        gt_track_graph,
        rec_track_graph,
        matching_threshold,
        sparse,
        validation_score,
        window_size,
        ignore_one_off_div_errors,
        fn_div_count_unconnected_parent):
    """Performs both matching and evaluation on the given
    gt and reconstructed tracks, and returns a Report
    with the results.

    Args
    ----
    gt_track_graph: linajea.tracking.TrackGraph
        Graph containing the ground truth annotations
    rec_track_graph: linajea.tracking.TrackGraph
        Reconstructed graph
    matching_threshold: int
        How far can a GT annotation and a predicted object be apart but
        still be matched to each other.
    sparse: bool
        Is the ground truth sparse (not every instance is annotated)
    validation_score: bool
        Should the validation score be computed (additional metric)
    window_size: int
        What is the maximum window size for which the fraction of
        error-free tracklets should be computed?
    ignore_one_off_div_errors: bool
        Division annotations are often slightly imprecise. Due to the
        limited temporal resolution the exact moment a division happens
        cannnot always be determined accuratly. If the predicted division
        happens 1 frame before or after an annotated one, does not count
        it as an error.
    fn_div_count_unconnected_parent: bool
        If the parent of the mother cell of a division is missing, should
        this count as a division error (aside from the already counted FN
        edge error)
    Returns
    -------
    Report
        Report object containing detailed result
    """
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
