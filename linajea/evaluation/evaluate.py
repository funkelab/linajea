from .match import match_edges
import logging

logger = logging.getLogger(__name__)


class Scores:

    def __repr__(self):

        return """\
EDGE STATISTICS
     num gt: %d
num matches: %d
        fps: %d
        fns: %d

TRACK STATISTICS
      num gt
 match/total: %d / %d
     num rec
 match/total: %d / %d
    edge fps
 in matched : %d
track breaks: %d
         """ % (
            # edge scores
            self.num_gt_edges,
            self.num_matched_edges,
            self.num_fp_edges,
            self.num_fn_edges,

            # track stats
            self.num_gt_matched_tracks,
            self.num_gt_tracks,
            self.num_rec_matched_tracks,
            self.num_rec_tracks,
            self.num_edge_fps_in_matched_tracks,
            self.num_track_breaks,
            )


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

    get_track_related_statistics(gt_track_graph, rec_track_graph, scores)
    return scores


def get_track_related_statistics(
        gt_track_graph,
        rec_track_graph,
        scores
        ):
    gt_tracks = gt_track_graph.get_tracks()
    rec_tracks = rec_track_graph.get_tracks()

    scores.num_gt_tracks = len(gt_tracks)
    scores.num_rec_tracks = len(rec_tracks)

    num_gt_matched_tracks = 0
    num_rec_matched_tracks = 0

    num_edge_fps_in_matched_tracks = 0
    num_track_breaks = 0

    edge_matches = scores.edge_matches
    x_to_y_edges = {x: y for x, y in edge_matches}
    y_to_x_edges = {y: x for x, y in edge_matches}

    edges_to_track_id_y = {}
    for index, track in enumerate(rec_tracks):
        for edge in track.edges():
            edges_to_track_id_y[edge] = index

    for gt_track in gt_tracks:
        match_index = -1
        matched = False
        for edge in gt_track.edges():
            if edge in x_to_y_edges.keys():
                matched = True
                match_edge = x_to_y_edges[edge]
                new_match_index = edges_to_track_id_y[match_edge]
                if match_index == -1:
                    match_index = new_match_index
                elif match_index != new_match_index:
                    num_track_breaks += 1
                    match_index = new_match_index
        if matched:
            num_gt_matched_tracks += 1

    for rec_track in rec_tracks:
        matched = False
        unmatched_edges = 0
        for edge in rec_track.edges():
            if edge in y_to_x_edges.keys():
                matched = True
            else:
                unmatched_edges += 1
        if matched:
            num_rec_matched_tracks += 1
            num_edge_fps_in_matched_tracks += unmatched_edges

    scores.num_gt_matched_tracks = num_gt_matched_tracks
    scores.num_rec_matched_tracks = num_rec_matched_tracks
    scores.num_edge_fps_in_matched_tracks = num_edge_fps_in_matched_tracks
    scores.num_track_breaks = num_track_breaks
