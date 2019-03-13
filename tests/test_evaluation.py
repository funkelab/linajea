import linajea.tracking
import linajea.evaluation as e
import logging
import unittest
import linajea
import daisy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('linajea.evaluation').setLevel(logging.DEBUG)


class EvaluationTestCase(unittest.TestCase):

    def get_tracks(self, cells, edges, roi):
        g = self.create_graph(cells, edges, roi)
        tracks = g.get_tracks()

        str_x = "\nTracks:\n"
        for track_id, track in enumerate(tracks):
            str_x += "Track %d has nodes %s and edges %s\n"\
                % (track_id, track.nodes, track.edges)
        logger.debug(str_x)
        return tracks

    def create_graph(self, cells, edges, roi):
        db = linajea.CandidateDatabase('test_eval', 'localhost')
        graph = db[roi]
        graph.add_nodes_from(cells)
        graph.add_edges_from(edges)
        tg = linajea.tracking.TrackGraph(graph_data=graph, frame_key='t')
        return tg

    def getTrack1(self):
        cells = [
                (1, {'t': 0, 'z': 0, 'y': 0, 'x': 0}),
                (2, {'t': 1, 'z': 0, 'y': 0, 'x': 0}),
                (3, {'t': 2, 'z': 0, 'y': 0, 'x': 0}),
                (4, {'t': 3, 'z': 0, 'y': 0, 'x': 0}),
            ]
        edges = [
            (2, 1),
            (3, 2),
            (4, 3)
            ]
        roi = daisy.Roi((0, 0, 0, 0), (4, 4, 4, 4))
        return cells, edges, roi

    def getDivisionTrack(self):
        cells = [
                (1, {'t': 0, 'z': 0, 'y': 0, 'x': 0}),
                (2, {'t': 1, 'z': 0, 'y': 0, 'x': 0}),
                (3, {'t': 2, 'z': 0, 'y': 0, 'x': 0}),
                (4, {'t': 3, 'z': 0, 'y': 0, 'x': 0}),
                (5, {'t': 2, 'z': 3, 'y': 0, 'x': 0}),
                (6, {'t': 3, 'z': 3, 'y': 0, 'x': 0}),
                (7, {'t': 4, 'z': 3, 'y': 0, 'x': 0}),
            ]
        edges = [
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 2),
            (6, 5),
            (7, 6),
            ]
        roi = daisy.Roi((0, 0, 0, 0), (5, 5, 5, 5))
        return cells, edges, roi

    def test_perfect_matching(self):
        cells, edges, roi = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        rec_tracks = self.get_tracks(cells, edges, roi)
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
        cells, edges, roi = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        rec_tracks = self.get_tracks(cells, edges, roi)

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
        cells, edges, roi = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        del edges[1]
        rec_tracks = self.get_tracks(cells, edges, roi)
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
        cells, edges, roi = self.getTrack1()
        gt_tracks = self.get_tracks(cells, edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        del edges[1]
        rec_tracks = self.get_tracks(cells, edges, roi)

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
        cells, edges, roi = self.getDivisionTrack()
        # introduce a false positive edge
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        del gt_cells[0]
        del gt_edges[0]
        gt_tracks = self.get_tracks(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        edges.remove((6, 5))
        rec_tracks = self.get_tracks(cells, edges, roi)
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
        cells, edges, roi = self.getDivisionTrack()
        # introduce a false positive edge
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        del gt_cells[0]
        del gt_edges[0]
        gt_tracks = self.get_tracks(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        edges.remove((6, 5))
        rec_tracks = self.get_tracks(cells, edges, roi)

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
        cells, edges, roi = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_edges.remove((5, 2))
        gt_tracks = self.get_tracks(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        rec_tracks = self.get_tracks(cells, edges, roi)
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
        cells, edges, roi = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_edges.remove((5, 2))
        gt_tracks = self.get_tracks(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        rec_tracks = self.get_tracks(cells, edges, roi)

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
