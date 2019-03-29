from .match import match_tracks
import logging

logger = logging.getLogger(__name__)


class Scores:

    def __init__(self):

        # the track matching scores
        self.num_gt_tracks = 0
        self.num_matched_tracks = 0
        self.num_splits = 0
        self.num_merges = 0
        self.num_fp_tracks = 0
        self.num_fn_tracks = 0

        # node scores
        self.num_gt_nodes = 0
        self.num_gt_nodes_in_matched_tracks = 0
        self.num_matched_nodes = 0
        self.num_fp_nodes = 0
        self.num_fn_nodes = 0

        # edge scores
        self.num_gt_edges = 0
        self.num_gt_edges_in_matched_tracks = 0
        self.num_matched_edges = 0
        self.num_fp_edges = 0
        self.num_fn_edges = 0
        self.num_edges = 0

        # division scores
        self.num_gt_divisions = 0
        self.num_gt_divisions_in_matched_tracks = 0
        self.num_matched_divisions = 0
        self.num_fp_divisions = 0
        self.num_fn_divisions = 0

    def __repr__(self):

        return """\
TRACK STATISTICS
     num gt: %d
num matches: %d
     splits: %d
     merges: %d
        fps: %d
        fns: %d

NODE STATISTICS
     num gt: %d
in matched
     tracks: %d
num matches: %d
        fps: %d
        fns: %d

EDGE STATISTICS
     num gt: %d
in matched
     tracks: %d
num matches: %d
        fps: %d
        fns: %d

DIVISION STATISTICS
     num gt: %d
in matched
     tracks: %d
num matches: %d
        fps: %d
        fns: %d
         """ % (
            self.num_gt_tracks,
            self.num_matched_tracks,
            self.num_splits,
            self.num_merges,
            self.num_fp_tracks,
            self.num_fn_tracks,

            # node scores
            self.num_gt_nodes,
            self.num_gt_nodes_in_matched_tracks,
            self.num_matched_nodes,
            self.num_fp_nodes,
            self.num_fn_nodes,

            # edge scores
            self.num_gt_edges,
            self.num_gt_edges_in_matched_tracks,
            self.num_matched_edges,
            self.num_fp_edges,
            self.num_fn_edges,

            # division scores
            self.num_gt_divisions,
            self.num_gt_divisions_in_matched_tracks,
            self.num_matched_divisions,
            self.num_fp_divisions,
            self.num_fn_divisions)


def evaluate(gt_tracks, rec_tracks, matching_threshold):

    logger.info("Matching GT tracks to REC track...")
    matched_tracks = match_tracks(
        gt_tracks,
        rec_tracks,
        matching_threshold)
    return evaluate_matches(matched_tracks, gt_tracks, rec_tracks)


def evaluate_matches(match_tracks_output, gt_tracks, rec_tracks):
    track_matches, cell_matches, s, m, fp, fn = match_tracks_output
    logger.debug("Track matches: %s" % track_matches)
    logger.debug("Cell matches: %s" % cell_matches)

    scores = Scores()
    # get ground truth statistics
    for track in gt_tracks:
        scores.num_gt_tracks += 1
        scores.num_gt_nodes += len(track.nodes)
        scores.num_gt_edges += len(track.edges)

    # store track statistics
    scores.num_splits = s
    scores.num_merges = m
    scores.num_fp_tracks = fp
    scores.num_fn_tracks = fn
    scores.num_matched_tracks = len(track_matches)

    # get node and edge statistics

    logger.info("Evaluating nodes and edges on matched tracks...")
    node_scores = evaluate_nodes(
        gt_tracks,
        rec_tracks,
        track_matches,
        cell_matches)

    edge_scores = evaluate_edges(
        gt_tracks,
        rec_tracks,
        track_matches,
        cell_matches)

    division_scores = evaluate_divisions(
        gt_tracks,
        rec_tracks,
        track_matches,
        cell_matches)
    scores.num_gt_nodes_in_matched_tracks = node_scores[0]
    scores.num_matched_nodes = node_scores[1]
    scores.num_fp_nodes = node_scores[2]
    scores.num_fn_nodes = node_scores[3]

    scores.num_gt_edges_in_matched_tracks = edge_scores[0]
    scores.num_matched_edges = edge_scores[1]
    scores.num_fp_edges = edge_scores[2]
    scores.num_fn_edges = edge_scores[3]

    scores.num_gt_divisions = division_scores[0]
    scores.num_gt_divisions_in_matched_tracks = division_scores[1]
    scores.num_matched_divisions = division_scores[2]
    scores.num_fp_divisions = division_scores[3]
    scores.num_fn_divisions = division_scores[4]

    return scores


def evaluate_nodes(gt_tracks, rec_tracks, track_matches, cell_matches):
    gt_matched_cells = [m[0] for m in cell_matches]
    rec_matched_cells = [m[1] for m in cell_matches]
    gt_matched_tracks = [m[0] for m in track_matches]
    rec_matched_tracks = [m[1] for m in track_matches]

    num_gt_nodes_in_matched_tracks = 0
    num_matches = 0
    num_fps = 0
    num_fns = 0

    for track_id, track in enumerate(gt_tracks):
        if track_id not in gt_matched_tracks:
            continue
        for node in track.nodes:
            num_gt_nodes_in_matched_tracks += 1
            if node in gt_matched_cells:
                num_matches += 1
            else:
                num_fns += 1
    for track_id, track in enumerate(rec_tracks):
        if track_id not in rec_matched_tracks:
            continue
        for node in track.nodes:
            if node not in rec_matched_cells:
                num_fps += 1
    return num_gt_nodes_in_matched_tracks, num_matches, num_fps, num_fns


def evaluate_edges(gt_tracks, rec_tracks, track_matches, cell_matches):
    gt_cells_to_rec = {m[0]: m[1] for m in cell_matches}
    rec_cells_to_gt = {m[1]: m[0] for m in cell_matches}
    gt_tracks_to_matched_edges = {}
    rec_tracks_to_matched_edges = {}
    for m in track_matches:
        gt_track = m[0]
        rec_track = m[1]
        if gt_track not in gt_tracks_to_matched_edges:
            gt_tracks_to_matched_edges[gt_track] = []
        gt_tracks_to_matched_edges[gt_track] += rec_tracks[rec_track].edges

        if rec_track not in rec_tracks_to_matched_edges:
            rec_tracks_to_matched_edges[rec_track] = []
        rec_tracks_to_matched_edges[rec_track] += gt_tracks[gt_track].edges

    num_gt_edges_in_matched_tracks = 0
    num_matches = 0
    num_fps = 0
    num_fns = 0

    for track_id, track in enumerate(gt_tracks):
        if track_id not in gt_tracks_to_matched_edges:
            continue
        for u, v in track.edges:
            num_gt_edges_in_matched_tracks += 1
            if u in gt_cells_to_rec and v in gt_cells_to_rec:
                match_u = gt_cells_to_rec[u]
                match_v = gt_cells_to_rec[v]
                if (match_u, match_v) in gt_tracks_to_matched_edges[track_id]:
                    logger.debug("Found edge match: gt (%d, %d), rec (%d, %d)"
                                 % (u, v, match_u, match_v))
                    num_matches += 1
                else:
                    logger.debug("Did not find rec edge (%d, %d) "
                                 "to match gt edge (%d, %d)"
                                 % (match_u, match_v, u, v))
                    num_fns += 1
            else:
                logger.debug("Did not find rec edge to match gt edge (%d, %d)"
                             % (u, v))
                num_fns += 1
    for track_id, track in enumerate(rec_tracks):
        if track_id not in rec_tracks_to_matched_edges:
            continue
        for u, v in track.edges:
            if u in rec_cells_to_gt and v in rec_cells_to_gt:
                match_u = rec_cells_to_gt[u]
                match_v = rec_cells_to_gt[v]
                matched_edges = rec_tracks_to_matched_edges[track_id]
                if (match_u, match_v) not in matched_edges:
                    num_fps += 1
            else:
                num_fps += 1

    return num_gt_edges_in_matched_tracks, num_matches, num_fps, num_fns


def evaluate_divisions(gt_tracks, rec_tracks, track_matches, cell_matches):
    gt_cells_to_rec = {m[0]: m[1] for m in cell_matches}
    rec_cells_to_gt = {m[1]: m[0] for m in cell_matches}
    gt_matched_tracks = [m[0] for m in track_matches]
    rec_matched_tracks = [m[1] for m in track_matches]

    num_gt_divisions = 0
    num_gt_divisions_in_matched_tracks = 0
    num_matches = 0
    num_fps = 0
    num_fns = 0

    gt_parents = []
    rec_parents = []

    for track_id, track in enumerate(gt_tracks):
        node_degrees = {node: degree for node, degree in track.in_degree()}
        logger.debug("Max degree for track %d: %d"
                     % (track_id, max(node_degrees.values())))
        assert max(node_degrees.values()) <= 2,\
            ("Max in degree should be less than 2, "
             "got %d in track %d"
             % (max(node_degrees.values()), track_id))
        parents = [node for node, degree in node_degrees.items()
                   if degree == 2]
        logger.debug("Parent nodes: %s" % parents)
        num_gt_divisions += len(parents)
        if track_id in gt_matched_tracks:
            gt_parents += parents

    for track_id, track in enumerate(rec_tracks):
        if track_id not in rec_matched_tracks:
            continue
        parents = [node for node in track.nodes if track.in_degree(node) == 2]
        rec_parents += parents

    num_gt_divisions_in_matched_tracks = len(gt_parents)
    for parent in gt_parents:
        if parent in gt_cells_to_rec and\
                gt_cells_to_rec[parent] in rec_parents:
            num_matches += 1
        else:
            num_fns += 1

    for parent in rec_parents:
        if parent not in rec_cells_to_gt or\
                rec_cells_to_gt[parent] not in gt_parents:
            num_fps += 1

    return (num_gt_divisions, num_gt_divisions_in_matched_tracks,
            num_matches, num_fps, num_fns)
