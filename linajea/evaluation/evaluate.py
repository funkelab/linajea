from .match import match_edges
import logging
from collections import deque
import math

logger = logging.getLogger(__name__)


class Scores:

    def __repr__(self):

        return """\
EDGE STATISTICS
     num gt: %d
num matches: %d
        fps: %d
        fns: %d
  precision: %f
     recall: %f
    f-score: %f
   f2-score: %f

TRACK STATISTICS
      num gt
 match/total: %d / %d
     num rec
 match/total: %d / %d
       AEFTL: %f
         ERL: %f
 identity_sw: %d

DIVISION STATISTICS
gt_divisions: %d
tp_divisions: %d
fp_divisions: %d
         """ % (
                 # edge scores
                 self.num_gt_edges,
                 self.num_matched_edges,
                 self.num_fp_edges,
                 self.num_fn_edges,
                 self.precision,
                 self.recall,
                 self.f_score,
                 self.f2_score,

                 # track stats
                 self.num_gt_matched_tracks,
                 self.num_gt_tracks,
                 self.num_rec_matched_tracks,
                 self.num_rec_tracks,
                 self.aeftl,
                 self.erl,
                 self.identity_switches,

                 # division stats
                 self.num_gt_divisions,
                 self.num_tp_divisions,
                 self.num_fp_divisions,
                 )


def evaluate(
        gt_track_graph,
        rec_track_graph,
        matching_threshold,
        error_details=False):

    logger.info("Matching GT edges to REC edges...")
    gt_edges, rec_edges, edge_matches, edge_fps = match_edges(
            gt_track_graph,
            rec_track_graph,
            matching_threshold)

    scores = Scores()
    scores.num_gt_edges = len(gt_edges)
    scores.num_matched_edges = len(edge_matches)
    scores.num_fp_edges = edge_fps
    scores.num_fn_edges = len(gt_edges) - len(edge_matches)
    scores.edge_matches = [(gt_edges[gt_ind], rec_edges[rec_ind])
                           for gt_ind, rec_ind in edge_matches]

    get_track_related_statistics(
            gt_track_graph, rec_track_graph, scores, error_details)
    add_f_score(scores)
    return scores


def get_track_related_statistics(
        x_track_graph,
        y_track_graph,
        scores,
        error_details=False
        ):
    logger.info("Getting track related statistics")
    edge_matches = scores.edge_matches

    # get tracks
    x_tracks = x_track_graph.get_tracks()
    y_tracks = y_track_graph.get_tracks()
    logger.debug("Found %d gt tracks and %d rec tracks"
                 % (len(x_tracks), len(y_tracks)))

    # get basic track stats
    scores.num_gt_tracks = len(x_tracks)
    scores.num_rec_tracks = len(y_tracks)

    # set starting values
    scores.num_gt_matched_tracks = 0
    scores.num_rec_matched_tracks = 0
    scores.num_tp_divisions = 0
    scores.num_fp_divisions = 0
    scores.identity_switches = 0
    reconstruction_lengths = []
    if error_details:
        scores.fn_edges = []
        scores.fp_division_nodes = []
        scores.tp_division_nodes = []
        scores.identity_switch_nodes = []

    # set up data structures
    edges_to_track_id_y = {}
    matched_y_edges = set()
    matched_y_track_ids = set()
    x_edges_to_y_edges = {}

    for index, track in enumerate(y_tracks):
        for edge in track.edges():
            edges_to_track_id_y[edge] = index

    for x_edge, y_edge in edge_matches:
        x_edges_to_y_edges[x_edge] = y_edge
        matched_y_edges.add(y_edge)
        matched_y_track_ids.add(edges_to_track_id_y[y_edge])

    for track in x_tracks:
        for x_edge in track.edges():
            if x_edge in x_edges_to_y_edges:
                scores.num_gt_matched_tracks += 1
                break

    scores.num_rec_matched_tracks = len(matched_y_track_ids)

    logger.info("Getting segment lengths")
    for x_track in x_tracks:
        logger.debug("Getting segments for track with nodes %s"
                     % list(x_track.nodes()))

        segment_lengths = []
        start_of_gt_track = True
        start_frame = x_track.get_frames()[0]
        start_cells = deque(x_track.cells_by_frame(start_frame))
        assert len(start_cells) == 1, "%d cells in start frame %d"\
            % (len(start_cells), start_frame)

        while len(start_cells) > 0:
            start_cell = start_cells.popleft()
            if start_cell not in x_track.nodes:
                continue
            next_edges = x_track.next_edges(start_cell)
            next_edges = deque([(edge, None) for edge in next_edges])
            length = 0

            while len(next_edges) > 0:
                next_edge, prev_cell_match = next_edges.pop()
                source, target = next_edge

                if next_edge not in x_edges_to_y_edges:
                    # false negative, no next edge in this segment
                    # add source (later cell) to start_cells
                    if error_details:
                        scores.fn_edges.append((int(next_edge[0]),
                                                int(next_edge[1])))
                    start_cells.append(source)
                    continue

                edge_match = x_edges_to_y_edges[next_edge]
                source_match, target_match = edge_match

                if prev_cell_match is not None and\
                        prev_cell_match != target_match:
                    # identity switch - no next edge in this segment
                    # add target (earlier cell) to start_cells
                    scores.identity_switches += 1
                    if error_details:
                        scores.identity_switch_nodes.append(int(target))
                    start_cells.append(target)
                    continue

                matched_track = y_tracks[edges_to_track_id_y[edge_match]]
                next_edges_in_matched_track = list(
                        matched_track.next_edges(target_match))

                if len(next_edges_in_matched_track) > 1:
                    # division in matched track
                    fp_division = True
                    assert edge_match in next_edges_in_matched_track
                    next_edges_in_matched_track.remove(edge_match)
                    assert len(next_edges_in_matched_track) == 1
                    other_edge_y = next_edges_in_matched_track[0]

                    if other_edge_y in matched_y_edges:
                        x_target_next_edges = list(x_track.next_edges(target))
                        if len(x_target_next_edges) != 1:
                            # division in ground truth track
                            assert len(x_target_next_edges) == 2
                            assert next_edge in x_target_next_edges
                            x_target_next_edges.remove(next_edge)
                            other_edge_x = x_target_next_edges[0]
                            if other_edge_x in x_edges_to_y_edges and\
                                    x_edges_to_y_edges[other_edge_x] ==\
                                    other_edge_y:
                                # true positive division
                                fp_division = False
                                # this is double counted by each child edge
                                scores.num_tp_divisions += 0.5
                                if error_details:
                                    scores.tp_division_nodes.append(
                                            int(target))

                    if fp_division and\
                            (start_of_gt_track or prev_cell_match is not None):
                        # false positive division, no next edge in this segment
                        # add target (earlier cell) to start_cells
                        scores.num_fp_divisions += 1
                        logger.debug("False positive division")
                        start_cells.append(target)
                        if error_details:
                            scores.fp_division_nodes.append(int(target))
                        continue

                # edge continues segment
                # add one to length of segment
                # add next edges to queue for this segment
                length += 1
                continuing_edges = x_track.next_edges(source)
                for cont_edge in continuing_edges:
                    next_edges.append((cont_edge, source_match))

            # no more edges in this segment
            if length > 0:
                segment_lengths.append(length)
            start_of_gt_track = False

        logger.debug("Found segment lengths %s" % segment_lengths)
        # add track stats to overall
        reconstruction_lengths.extend(segment_lengths)

    logger.debug("Segment lengths: %s" % reconstruction_lengths)
    scores.aeftl = float(
        sum(reconstruction_lengths)) / len(reconstruction_lengths)
    scores.erl = sum(map(lambda b: math.pow(b, 2),
                         reconstruction_lengths)) / scores.num_gt_edges
    # division stats
    scores.num_gt_divisions = 0
    for track_id, track in enumerate(x_tracks):
        node_degrees = track.in_degree()
        max_node_degree = max([v for _, v in node_degrees])
        logger.debug("Max degree for track %d: %d"
                     % (track_id, max_node_degree))
        assert max_node_degree <= 2,\
            ("Max in degree should be less than 2, "
             "got %d in track %d"
             % (max_node_degree, track_id))
        parents = [node for node, degree in node_degrees
                   if degree == 2]
        logger.debug("Parent nodes: %s" % parents)
        scores.num_gt_divisions += len(parents)


def add_f_score(scores):
    tp = scores.num_matched_edges
    fp = scores.num_fp_edges
    fn = scores.num_fn_edges

    scores.precision = tp / (tp + fp)
    scores.recall = tp / (tp + fn)
    scores.f_score = 2 * scores.precision * scores.recall / (
            scores.precision + scores.recall)
    scores.f2_score = 5 * scores.precision * scores.recall / (
            4 * scores.precision + scores.recall)
