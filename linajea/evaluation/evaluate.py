from .match import match_edges
import logging

logger = logging.getLogger(__name__)


class Scores:

    def __init__(self):

        # edge scores
        self.num_gt_edges = 0
        self.num_fp_edges = 0
        self.num_fn_edges = 0
        self.num_edges = 0
        self.edge_matches = []

    def __repr__(self):

        return """\
EDGE STATISTICS
     num gt: %d
num matches: %d
        fps: %d
        fns: %d
         """ % (
            # edge scores
            self.num_gt_edges,
            self.num_matched_edges,
            self.num_fp_edges,
            self.num_fn_edges)


def evaluate(gt_track_graph, rec_track_graph, matching_threshold):

    logger.info("Matching GT tracks to REC track...")
    gt_edges, rec_edges, edge_matches = match_edges(
        gt_track_graph,
        rec_track_graph,
        matching_threshold)

    scores = Scores()
    scores.num_gt_edges = len(gt_edges)
    scores.num_matched_edges = len(edge_matches)
    scores.num_fp_edges = len(rec_edges) - len(edge_matches)
    scores.num_fn_edges = len(gt_edges) - len(edge_matches)
    scores.edge_matches = [(gt_edges[gt_ind], rec_edges[rec_ind])
                           for gt_ind, rec_ind in edge_matches]
    return scores
