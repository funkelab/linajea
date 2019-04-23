from .match import match_edges
import logging
import networkx as nx

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
    edge fps
 in matched : %d
 avg segment: %f
 identity sw: %d
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
            self.num_edge_fps_in_matched_tracks,
            self.avg_segment_length,
            self.num_identity_switches,
            )


def evaluate(gt_track_graph, rec_track_graph, matching_threshold):

    logger.info("Matching GT edges to REC edges...")
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
    add_f_score(scores)
    return scores


def get_track_related_statistics(
        x_track_graph,
        y_track_graph,
        scores
        ):
    logger.info("Getting track related statistics")
    edge_matches = scores.edge_matches

    # get tracks
    x_tracks = x_track_graph.get_tracks()
    y_tracks = y_track_graph.get_tracks()

    # get basic track stats
    scores.num_gt_tracks = len(x_tracks)
    scores.num_rec_tracks = len(y_tracks)

    # set starting values
    scores.num_gt_matched_tracks = 0
    scores.num_rec_matched_tracks = 0
    scores.num_edge_fps_in_matched_tracks = 0
    scores.num_identity_switches = 0
    reconstruction_lengths = []

    # add y track label to x edges
    logger.info("Adding matched track label to gt edges")
    edges_to_track_id_y = {}
    for index, track in enumerate(y_tracks):
        for edge in track.edges():
            edges_to_track_id_y[edge] = index
    x_edges_to_y_track_ids = {}
    for x_edge, y_edge in edge_matches:
        x_edges_to_y_track_ids[x_edge] = edges_to_track_id_y[y_edge]

    for track in x_tracks:
        matched = False
        for edge in track.edges():
            if edge in x_edges_to_y_track_ids:
                label = x_edges_to_y_track_ids[edge]
                matched = True
            else:
                label = -1
            track.edges[edge]['y_track_id'] = label
        if matched:
            scores.num_gt_matched_tracks += 1

    matched_y_edges = set(y for x, y in edge_matches)

    logger.info("Getting identity switches and segment lengths")
    for x_track in x_tracks:
        num_identity_switches = get_identity_switches(x_track)
        segment_lengths = get_segment_lengths(x_track)
        scores.num_identity_switches += num_identity_switches
        reconstruction_lengths.extend(segment_lengths)
    scores.avg_segment_length = float(
            sum(reconstruction_lengths)) / len(reconstruction_lengths)

    logger.info("Getting information about rec matched tracks")
    for y_track_index in set(x_edges_to_y_track_ids.values()):
        y_track = y_tracks[y_track_index]
        unmatched_edges = 0
        for edge in y_track.edges():
            if edge not in matched_y_edges:
                unmatched_edges += 1
        scores.num_rec_matched_tracks += 1
        scores.num_edge_fps_in_matched_tracks += unmatched_edges


def get_identity_switches(track):
    frames = track.get_frames()
    start_cell = track.cells_by_frame(frames[0])
    assert len(start_cell) == 1
    start_cell = start_cell[0]
    next_edges = list(track.next_edges(start_cell))
    return get_switches_helper(track, next_edges, None)


def get_switches_helper(track, next_edges, prev_y_track_id):
    num_identity_switches = 0

    while len(next_edges) == 1:
        next_edge = next_edges[0]
        y_track_id = track.edges[next_edge]['y_track_id']

        if y_track_id == -1:
            child = next_edge[0]
            next_edges = list(track.next_edges(child))
            continue
        elif prev_y_track_id is not None and prev_y_track_id != y_track_id:
            num_identity_switches += 1
        child = next_edge[0]
        next_edges = list(track.next_edges(child))
        prev_y_track_id = y_track_id

    if len(next_edges) == 0:
        return num_identity_switches
    elif len(next_edges) == 2:
        # special case: nothing is matched before the division
        if prev_y_track_id is None and\
                track.edges[next_edges[0]]['y_track_id'] !=\
                track.edges[next_edges[1]]['y_track_id']:
            assert num_identity_switches == 0
            num_identity_switches = 1
        return num_identity_switches +\
            get_switches_helper(track, [next_edges[0]], prev_y_track_id) +\
            get_switches_helper(track, [next_edges[1]], prev_y_track_id)


def add_f_score(scores):
    tp = scores.num_matched_edges
    fp = scores.num_edge_fps_in_matched_tracks
    fn = scores.num_fn_edges

    scores.precision = tp / (tp + fp)
    scores.recall = tp / (tp + fn)
    scores.f_score = 2 * scores.precision * scores.recall / (
                     scores.precision + scores.recall)
    scores.f2_score = 5 * scores.precision * scores.recall / (
                     4 * scores.precision + scores.recall)


def get_segment_lengths(x_track):
    lengths = []
    subgraphs = {}
    for u, v, data in x_track.edges(data=True):
        label = data['y_track_id']
        if label != -1:
            if label not in subgraphs.keys():
                subgraphs[label] = nx.Graph()
            subgraphs[label].add_edge(u, v)

    for subgraph in subgraphs.values():
        for c in nx.connected_components(subgraph):
            component = subgraph.subgraph(c)
            lengths.append(component.number_of_edges())

    return lengths
