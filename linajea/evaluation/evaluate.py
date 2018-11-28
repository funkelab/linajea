from .match import match_tracks
import logging

logger = logging.getLogger(__name__)

class Scores:

    def __init__(self):

        # the track matching scores
        self.num_splits = 0
        self.num_merges = 0
        self.num_fp_tracks = 0
        self.num_fn_tracks = 0
        self.num_tracks = 0
        self.num_matches = 0

        # edge scores
        self.num_fp_edges = 0
        self.num_fn_edges = 0
        self.num_edges = 0
        self.num_fp_divisions = 0
        self.num_fn_divisions = 0
        self.num_divisions = 0

    def __repr__(self):

        return """\
track splits: %d
      merges: %d
         fps: %d
         fns: %d
         num gt tracks: %d
         num matches: %d

edge     fps: %d
         fns: %d
         num: %d

division fps: %d
         fns: %d
         num: %d"""%(
            self.num_splits,
            self.num_merges,
            self.num_fp_tracks,
            self.num_fn_tracks,
            self.num_tracks,
            self.num_matches,
            self.num_fp_edges,
            self.num_fn_edges,
            self.num_edges,
            self.num_fp_divisions,
            self.num_fn_divisions,
            self.num_divisions)

def evaluate(gt_tracks, rec_tracks, matching_threshold):

    logger.info("Matching GT tracks to REC track...")
    track_matches, cell_matches, s, m, fp, fn = match_tracks(
        gt_tracks,
        rec_tracks,
        matching_threshold)

    scores = Scores()
    scores.num_splits = s
    scores.num_merges = m
    scores.num_fp_tracks = fp
    scores.num_fn_tracks = fn
    scores.num_tracks = len(gt_tracks)
    scores.num_matches = len(track_matches)

    logger.info("Evaluating edges on matched tracks...")
    fpe, fne, nd, fpd, fnd = evaluate_edges(
        gt_tracks,
        rec_tracks,
        track_matches,
        cell_matches)

    scores.num_fp_edges = fpe
    scores.num_fn_edges = fne
    scores.num_edges = sum([ len(track.edges) for track in gt_tracks ])
    scores.num_fp_divisions = fpd
    scores.num_fn_divisions = fnd
    scores.num_divisions = nd

    return scores

def evaluate_edges(gt_tracks, rec_tracks, track_matches, cell_matches):

    gt_to_rec_cell = { m[0]: m[1] for m in cell_matches }
    rec_to_gt_cell = { m[1]: m[0] for m in cell_matches }

    cell_to_rec_track = {
        c: track_id
        for track_id, track in enumerate(rec_tracks)
        for c in track.nodes
    }
    cell_to_gt_track = {
        c: track_id
        for track_id, track in enumerate(gt_tracks)
        for c in track.nodes
    }

    fp_edges = count_edge_fns(rec_tracks, rec_to_gt_cell, cell_to_gt_track)
    fn_edges = count_edge_fns(gt_tracks, gt_to_rec_cell, cell_to_rec_track)

    _, fp_divisions = count_division_fns(
        rec_tracks,
        rec_to_gt_cell,
        cell_to_gt_track)
    num_divisions, fn_divisions = count_division_fns(
        gt_tracks,
        gt_to_rec_cell,
        cell_to_rec_track)

    return fp_edges, fn_edges, num_divisions, fp_divisions, fn_divisions

def count_edge_fns(tracks_x, x_to_y_cell, cell_to_track_y):

    fn_edges = 0

    for track_x in tracks_x:
        for edge in track_x.edges:

            source_x, target_x = edge

            # either not matched
            if source_x not in x_to_y_cell or target_x not in x_to_y_cell:
                fn_edges += 1
                continue

            source_y, target_y = (
                x_to_y_cell[source_x],
                x_to_y_cell[target_x])

            # or matched to two different tracks
            if cell_to_track_y[source_y] != cell_to_track_y[target_y]:
                fn_edges += 1

    return fn_edges

def count_division_fns(tracks_x, x_to_y_cell, cell_to_track_y):

    num_divisions = 0
    fn_divisions = 0

    for track_x in tracks_x:
        for cell_x in track_x.nodes:

            # only divisions
            if track_x.degree(cell_x) != 3:
                continue

            frame = track_x.nodes[cell_x]['frame']

            children_x = [
                n
                for n in track_x.neighbors(cell_x)
                if track_x.nodes[n]['frame'] == frame + 1
            ]

            if len(children_x) != 2:
                logger.error(
                    "There is a weird cell (%d) with three neighbors that "
                    "does not look like a division", cell_x)
                continue

            num_divisions += 1

            # either not matched
            if (
                    cell_x not in x_to_y_cell or
                    children_x[0] not in x_to_y_cell or
                    children_x[1] not in x_to_y_cell):
                fn_divisions += 1
                continue

            tracks_y = set([
                cell_to_track_y[x_to_y_cell[cell_x]],
                cell_to_track_y[x_to_y_cell[children_x[0]]],
                cell_to_track_y[x_to_y_cell[children_x[1]]]
            ])

            # or matched to several different tracks
            if len(tracks_y) != 1:
                fn_divisions += 1

    return num_divisions, fn_divisions
