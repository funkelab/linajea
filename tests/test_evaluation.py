import linajea.tracking
import linajea.evaluation as e
import logging
import unittest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('linajea.evaluation').setLevel(logging.DEBUG)


class EvaluationTestCase(unittest.TestCase):

    def get_tracks(self, cells, edges):
        g = self.create_graph(cells, edges)
        tracks = g.get_tracks()

        str_x = "\nTracks:\n"
        for track_id, track in enumerate(tracks):
            str_x += "Track %d has nodes %s and edges %s\n"\
                % (track_id, track.nodes, track.edges)
        logger.debug(str_x)
        return tracks

    def create_graph(self, cells, edges):
        g = linajea.tracking.TrackGraph()
        for cell in cells:
            g.add_cell(cell)
        for edge in edges:
            g.add_cell_edge(edge)
        return g

    def getTrack1(self):
        cells = [
                {'id': 1, 'position': [0, 0, 0, 0], 'frame': 0},
                {'id': 2, 'position': [1, 0, 0, 0], 'frame': 1},
                {'id': 3, 'position': [2, 0, 0, 0], 'frame': 2},
                {'id': 4, 'position': [3, 0, 0, 0], 'frame': 3}
            ]
        edges = [
            {'source': 2, 'target': 1},
            {'source': 3, 'target': 2},
            {'source': 4, 'target': 3}
            ]
        return cells, edges

    def getDivisionTrack(self):
        cells = [
                {'id': 1, 'position': [0, 0, 0, 0], 'frame': 0},
                {'id': 2, 'position': [1, 0, 0, 0], 'frame': 1},
                {'id': 3, 'position': [2, 0, 0, 0], 'frame': 2},
                {'id': 4, 'position': [3, 0, 0, 0], 'frame': 3},
                {'id': 5, 'position': [2, 3, 0, 0], 'frame': 2},
                {'id': 6, 'position': [3, 3, 0, 0], 'frame': 3},
                {'id': 7, 'position': [4, 3, 0, 0], 'frame': 4}
            ]
        edges = [
            {'source': 2, 'target': 1},
            {'source': 3, 'target': 2},
            {'source': 4, 'target': 3},
            {'source': 5, 'target': 2},
            {'source': 6, 'target': 5},
            {'source': 7, 'target': 6}
            ]
        return cells, edges

    def test_perfect_matching(self):
        cells, edges = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges)
        for cell in cells:
            cell['position'][2] += 1
        rec_tracks = self.get_tracks(cells, edges)
        match_output = e.match_tracks(gt_tracks, rec_tracks, 2)
        track_matches, cell_matches, splits, merges, fp, fn = match_output

        expected_track_matches = [(0, 0)]
        expected_cell_matches = [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4)
                ]

        self.assertCountEqual(track_matches, expected_track_matches)
        self.assertCountEqual(cell_matches, expected_cell_matches)
        self.assertEqual(splits, 0)
        self.assertEqual(merges, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_perfect_evaluation(self):
        cells, edges = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges)
        for cell in cells:
            cell['position'][2] += 1
        rec_tracks = self.get_tracks(cells, edges)

        scores = e.evaluate(gt_tracks, rec_tracks, 2)
        self.assertEqual(scores.num_gt_tracks, 1)
        self.assertEqual(scores.num_matched_tracks, 1)
        self.assertEqual(scores.num_splits, 0)
        self.assertEqual(scores.num_merges, 0)
        self.assertEqual(scores.num_fp_tracks, 0)
        self.assertEqual(scores.num_fn_tracks, 0)

        self.assertEqual(scores.num_gt_nodes, 4)
        self.assertEqual(scores.num_matched_nodes, 4)
        self.assertEqual(scores.num_fp_nodes, 0)
        self.assertEqual(scores.num_fn_nodes, 0)

        self.assertEqual(scores.num_gt_edges, 3)
        self.assertEqual(scores.num_matched_edges, 3)
        self.assertEqual(scores.num_fp_edges, 0)
        self.assertEqual(scores.num_fn_edges, 0)

        self.assertEqual(scores.num_gt_divisions, 0)
        self.assertEqual(scores.num_matched_divisions, 0)
        self.assertEqual(scores.num_fp_divisions, 0)
        self.assertEqual(scores.num_fn_divisions, 0)

    def test_imperfect_matching(self):
        cells, edges = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges)
        for cell in cells:
            cell['position'][2] += 1
        # introduce a split error
        del edges[1]
        rec_tracks = self.get_tracks(cells, edges)
        match_output = e.match_tracks(gt_tracks, rec_tracks, 2)
        track_matches, cell_matches, splits, merges, fp, fn = match_output

        expected_track_matches = [(0, 0), (0, 1)]
        expected_cell_matches = [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4)
                ]

        self.assertCountEqual(track_matches, expected_track_matches)
        self.assertCountEqual(cell_matches, expected_cell_matches)
        self.assertEqual(splits, 1)
        self.assertEqual(merges, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_imperfect_evaluation(self):
        cells, edges = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges)
        for cell in cells:
            cell['position'][2] += 1
        # introduce a split error
        del edges[1]
        rec_tracks = self.get_tracks(cells, edges)

        scores = e.evaluate(gt_tracks, rec_tracks, 2)
        self.assertEqual(scores.num_gt_tracks, 1)
        self.assertEqual(scores.num_matched_tracks, 2)
        self.assertEqual(scores.num_splits, 1)
        self.assertEqual(scores.num_merges, 0)
        self.assertEqual(scores.num_fp_tracks, 0)
        self.assertEqual(scores.num_fn_tracks, 0)

        self.assertEqual(scores.num_gt_nodes, 4)
        self.assertEqual(scores.num_matched_nodes, 4)
        self.assertEqual(scores.num_fp_nodes, 0)
        self.assertEqual(scores.num_fn_nodes, 0)

        self.assertEqual(scores.num_gt_edges, 3)
        self.assertEqual(scores.num_matched_edges, 2)
        self.assertEqual(scores.num_fp_edges, 0)
        self.assertEqual(scores.num_fn_edges, 1)

        self.assertEqual(scores.num_gt_divisions, 0)
        self.assertEqual(scores.num_matched_divisions, 0)
        self.assertEqual(scores.num_fp_divisions, 0)
        self.assertEqual(scores.num_fn_divisions, 0)

    def test_division_matching(self):
        cells, edges = self.getDivisionTrack()
        # introduce a false positive edge
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        del gt_cells[0]
        del gt_edges[0]
        gt_tracks = self.get_tracks(gt_cells, gt_edges)
        for cell in cells:
            cell['position'][2] += 1
        # introduce a split error
        edges.remove({'source': 6, 'target': 5})
        rec_tracks = self.get_tracks(cells, edges)
        match_output = e.match_tracks(gt_tracks, rec_tracks, 2)
        track_matches, cell_matches, splits, merges, fp, fn = match_output

        expected_track_matches = [(0, 0), (0, 1)]
        expected_cell_matches = [
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7)
                ]

        self.assertCountEqual(track_matches, expected_track_matches)
        self.assertCountEqual(cell_matches, expected_cell_matches)
        self.assertEqual(splits, 1)
        # self.assertEqual(merges, 0) with quadmatch and my intuition
        self.assertEqual(merges, 1)  # with comatch and Jan's intuition
        # self.assertEqual(fp, 0) with quadmatch and my intuition
        self.assertEqual(fp, 1)  # with comatch
        self.assertEqual(fn, 0)

    def test_division_evaluation(self):
        cells, edges = self.getDivisionTrack()
        # introduce a false positive edge
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        del gt_cells[0]
        del gt_edges[0]
        gt_tracks = self.get_tracks(gt_cells, gt_edges)
        for cell in cells:
            cell['position'][2] += 1
        # introduce a split error
        edges.remove({'source': 6, 'target': 5})
        rec_tracks = self.get_tracks(cells, edges)

        scores = e.evaluate(gt_tracks, rec_tracks, 2)
        self.assertEqual(scores.num_gt_tracks, 1)
        self.assertEqual(scores.num_matched_tracks, 2)
        self.assertEqual(scores.num_splits, 1)
        # self.assertEqual(scores.num_merges, 0) with quadmatch/my intuition
        self.assertEqual(scores.num_merges, 1)  # with comatch
        # self.assertEqual(scores.num_fp_tracks, 0) with quadmatch/my intuition
        self.assertEqual(scores.num_fp_tracks, 1)  # with comatch
        self.assertEqual(scores.num_fn_tracks, 0)

        self.assertEqual(scores.num_gt_nodes, 6)
        self.assertEqual(scores.num_matched_nodes, 6)
        self.assertEqual(scores.num_fp_nodes, 1)
        self.assertEqual(scores.num_fn_nodes, 0)

        self.assertEqual(scores.num_gt_edges, 5)
        self.assertEqual(scores.num_matched_edges, 4)
        self.assertEqual(scores.num_fp_edges, 1)
        self.assertEqual(scores.num_fn_edges, 1)

        self.assertEqual(scores.num_gt_divisions, 1)
        self.assertEqual(scores.num_matched_divisions, 1)
        self.assertEqual(scores.num_fp_divisions, 0)
        self.assertEqual(scores.num_fn_divisions, 0)

    def test_false_division_matching(self):
        cells, edges = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_edges.remove({'source': 5, 'target': 2})
        gt_tracks = self.get_tracks(gt_cells, gt_edges)
        for cell in cells:
            cell['position'][2] += 1
        rec_tracks = self.get_tracks(cells, edges)
        match_output = e.match_tracks(gt_tracks, rec_tracks, 2)
        track_matches, cell_matches, splits, merges, fp, fn = match_output

        expected_track_matches = [(0, 0), (1, 0)]
        expected_cell_matches = [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7)
                ]

        self.assertCountEqual(track_matches, expected_track_matches)
        self.assertCountEqual(cell_matches, expected_cell_matches)
        self.assertEqual(splits, 0)
        self.assertEqual(merges, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_false_division_evaluation(self):
        cells, edges = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_edges.remove({'source': 5, 'target': 2})
        gt_tracks = self.get_tracks(gt_cells, gt_edges)
        for cell in cells:
            cell['position'][2] += 1
        rec_tracks = self.get_tracks(cells, edges)

        scores = e.evaluate(gt_tracks, rec_tracks, 2)
        self.assertEqual(scores.num_gt_tracks, 2)
        self.assertEqual(scores.num_matched_tracks, 2)
        self.assertEqual(scores.num_splits, 0)
        self.assertEqual(scores.num_merges, 1)
        self.assertEqual(scores.num_fp_tracks, 0)
        self.assertEqual(scores.num_fn_tracks, 0)

        self.assertEqual(scores.num_gt_nodes, 7)
        self.assertEqual(scores.num_matched_nodes, 7)
        self.assertEqual(scores.num_fp_nodes, 0)
        self.assertEqual(scores.num_fn_nodes, 0)

        self.assertEqual(scores.num_gt_edges, 5)
        self.assertEqual(scores.num_matched_edges, 5)
        self.assertEqual(scores.num_fp_edges, 1)
        self.assertEqual(scores.num_fn_edges, 0)

        self.assertEqual(scores.num_gt_divisions, 0)
        self.assertEqual(scores.num_matched_divisions, 0)
        self.assertEqual(scores.num_fp_divisions, 1)
        self.assertEqual(scores.num_fn_divisions, 0)
