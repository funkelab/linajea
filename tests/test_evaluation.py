import logging
import unittest

import networkx as nx

import daisy

import linajea.evaluation as e
import linajea.tracking
import linajea.utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('linajea.evaluation').setLevel(logging.DEBUG)


class EvaluationTestCase(unittest.TestCase):

    def create_graph(self, cells, edges, roi):
        graph = nx.DiGraph()
        graph.add_nodes_from(cells)
        graph.add_edges_from(edges)
        tg = linajea.tracking.TrackGraph(graph_data=graph, frame_key='t',
                                         roi=roi)
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

    def test_perfect_evaluation(self):
        cells, edges, roi = self.getTrack1()
        gt_track_graph = self.create_graph(cells, edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)
        self.assertEqual(scores.matched_edges, 3)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 0)
        self.assertEqual(scores.gt_tracks, 1)
        self.assertEqual(scores.gt_matched_tracks, 1)
        self.assertEqual(scores.rec_matched_tracks, 1)
        self.assertEqual(scores.rec_tracks, 1)
        self.assertEqual(scores.aeftl, 3)
        self.assertEqual(scores.erl, 3)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 0)
        self.assertEqual(scores.fp_divisions, 0)
        self.assertEqual(scores.fn_divisions, 0)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 1.0)
        self.assertAlmostEqual(scores.f_score, 1.0)
        self.assertEqual(scores.correct_segments["1"], (3, 3))
        self.assertEqual(scores.correct_segments["2"], (2, 2))
        self.assertEqual(scores.correct_segments["3"], (1, 1))

    def test_imperfect_evaluation(self):
        cells, edges, roi = self.getTrack1()
        gt_track_graph = self.create_graph(cells, edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        del edges[1]
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)
        self.assertEqual(scores.matched_edges, 2)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 1)
        self.assertEqual(scores.gt_tracks, 1)
        self.assertEqual(scores.gt_matched_tracks, 1)
        self.assertEqual(scores.rec_matched_tracks, 2)
        self.assertEqual(scores.rec_tracks, 2)
        self.assertEqual(scores.aeftl, 1)
        self.assertAlmostEqual(scores.erl, 2./3)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 0)
        self.assertEqual(scores.fn_divisions, 0)
        self.assertEqual(scores.fp_divisions, 0)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 2./3)
        self.assertAlmostEqual(scores.f_score, 4./5)
        self.assertEqual(scores.correct_segments["1"], (2, 3))
        self.assertEqual(scores.correct_segments["2"], (0, 2))
        self.assertEqual(scores.correct_segments["3"], (0, 1))

    def test_fn_division_evaluation(self):
        cells, edges, roi = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        edges.remove((5, 2))
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True,
            validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)

        self.assertEqual(scores.matched_edges, 5)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 1)
        self.assertEqual(scores.gt_tracks, 1)
        self.assertEqual(scores.gt_matched_tracks, 1)
        self.assertEqual(scores.rec_matched_tracks, 2)
        self.assertEqual(scores.rec_tracks, 2)
        self.assertAlmostEqual(scores.aeftl, 2.5)
        self.assertAlmostEqual(scores.erl, 13./6)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 1)
        self.assertEqual(scores.fn_divisions, 1)
        self.assertEqual(scores.fp_divisions, 0)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 5./6)
        self.assertEqual(scores.correct_segments["1"], (5, 6))
        self.assertEqual(scores.correct_segments["2"], (2, 4))
        self.assertEqual(scores.correct_segments["3"], (0, 2))
        self.assertEqual(scores.correct_segments["4"], (0, 1))

    def test_fn_division_evaluation2(self):
        cells, edges, roi = self.getDivisionTrack()
        # introduce a false positive edge
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        del gt_cells[0]
        del gt_edges[0]
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        edges.remove((5, 2))
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)
        self.assertEqual(scores.matched_edges, 4)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 1)
        self.assertEqual(scores.gt_tracks, 1)
        self.assertEqual(scores.gt_matched_tracks, 1)
        self.assertEqual(scores.rec_matched_tracks, 2)
        self.assertEqual(scores.rec_tracks, 2)
        self.assertAlmostEqual(scores.aeftl, 2.)
        self.assertAlmostEqual(scores.erl, 8./5)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 1)
        self.assertEqual(scores.fn_divisions, 1)
        self.assertEqual(scores.fp_divisions, 0)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 4./5)
        self.assertEqual(scores.correct_segments["1"], (4, 5))
        self.assertEqual(scores.correct_segments["2"], (2, 3))
        self.assertEqual(scores.correct_segments["3"], (0, 1))

    def test_fn_division_evaluation3(self):
        cells, edges, roi = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        # introduce a split error
        edges.remove((5, 2))
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)

        self.assertEqual(scores.matched_edges, 5)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 1)
        self.assertEqual(scores.gt_tracks, 1)
        self.assertEqual(scores.gt_matched_tracks, 1)
        self.assertEqual(scores.rec_matched_tracks, 2)
        self.assertEqual(scores.rec_tracks, 2)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 1)
        self.assertEqual(scores.fn_divisions, 1)
        self.assertEqual(scores.fp_divisions, 0)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 5./6)

    def test_fp_division_evaluation(self):
        cells, edges, roi = self.getDivisionTrack()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        # remove division from gt
        gt_edges.remove((5, 2))
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)
        for cell in cells:
            cell[1]['y'] += 1
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)

        self.assertEqual(scores.matched_edges, 5)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 0)
        self.assertEqual(scores.gt_tracks, 2)
        self.assertEqual(scores.gt_matched_tracks, 2)
        self.assertEqual(scores.rec_matched_tracks, 1)
        self.assertEqual(scores.rec_tracks, 1)
        self.assertAlmostEqual(scores.aeftl, 5./3)
        self.assertAlmostEqual(scores.erl, 9./5)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 0)
        self.assertEqual(scores.fn_divisions, 0)
        self.assertEqual(scores.fp_divisions, 1)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 1.0)
        self.assertEqual(scores.correct_segments["1"], (5, 5))
        self.assertEqual(scores.correct_segments["2"], (2, 3))
        self.assertEqual(scores.correct_segments["3"], (0, 1))

    def test_fp_division_evaluation_at_beginning_of_gt(self):
        cells, edges, roi = self.getTrack1()
        gt_cells = cells.copy()
        gt_edges = edges.copy()
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)
        # add split at beginning of reconstruction
        for cell in cells:
            cell[1]['y'] += 1
        cells.append(
                (5, {'t': 1, 'z': 2, 'y': 1, 'x': 0})
                )
        edges.append((5, 1))
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)

        self.assertEqual(scores.matched_edges, 3)
        self.assertEqual(scores.fp_edges, 0)
        self.assertEqual(scores.fn_edges, 0)
        self.assertEqual(scores.gt_tracks, 1)
        self.assertEqual(scores.gt_matched_tracks, 1)
        self.assertEqual(scores.rec_matched_tracks, 1)
        self.assertEqual(scores.rec_tracks, 1)
        self.assertAlmostEqual(scores.aeftl, 3)
        self.assertAlmostEqual(scores.erl, 3)
        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 0)
        self.assertEqual(scores.fn_divisions, 0)
        self.assertEqual(scores.fp_divisions, 1)
        self.assertAlmostEqual(scores.precision, 1.0)
        self.assertAlmostEqual(scores.recall, 1.0)

    def test_one_off_fp_division_evaluation(self):
        roi = daisy.Roi((0, 0, 0, 0), (5, 5, 5, 5))
        gt_cells = [
                (251, {'t': 0, 'z': 0, 'y': 0, 'x': 0}),
                (261, {'t': 1, 'z': 0, 'y': 0, 'x': 0}),
                (271, {'t': 2, 'z': 0, 'y': 0, 'x': 0}),
                (281, {'t': 3, 'z': 0, 'y': 0, 'x': 0}),
                (282, {'t': 3, 'z': 3, 'y': 0, 'x': 0}),
                (291, {'t': 4, 'z': 0, 'y': 0, 'x': 0}),
                (301, {'t': 5, 'z': 0, 'y': 0, 'x': 0}),
                (302, {'t': 5, 'z': -3, 'y': 0, 'x': 0}),
            ]
        gt_edges = [
            (261, 251),
            (271, 261),
            (281, 271),
            (282, 271),
            (291, 281),
            (301, 291),
            (302, 291),
            ]
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)

        cells = [
                (2297, {'t': 0, 'z': 0, 'y': 0, 'x': 0}),
                (2788, {'t': 1, 'z': 0, 'y': 0, 'x': 0}),
                (2944, {'t': 2, 'z': 0, 'y': 0, 'x': 0}),
                (2453, {'t': 3, 'z': 0, 'y': 0, 'x': 0}),
                (3108, {'t': 2, 'z': 1, 'y': 0, 'x': 0}),
                (2990, {'t': 3, 'z': 3, 'y': 0, 'x': 0}),
                (2497, {'t': 4, 'z': 0, 'y': 0, 'x': 0}),
                (2498, {'t': 5, 'z': 0, 'y': 0, 'x': 0}),
                (2499, {'t': 5, 'z': -3, 'y': 0, 'x': 0}),
            ]
        edges = [
            (2788, 2297),
            (2944, 2788),
            (3108, 2788),
            (2990, 3108),
            (2453, 2944),
            (2497, 2453),
            (2498, 2497),
            (2499, 2497),
            ]
        for cell in cells:
            cell[1]['y'] += 1
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)

        self.assertEqual(scores.identity_switches, 0)
        self.assertEqual(scores.gt_divisions, 2)
        self.assertEqual(scores.fn_divisions, 1)
        self.assertEqual(scores.fp_divisions, 1)
        self.assertEqual(scores.fn_edges, 0)

    def test_one_off_fp_division_evaluation2(self):
        roi = daisy.Roi((0, 0, 0, 0), (10, 5, 5, 5))
        gt_cells = [
                (251, {'t': 0, 'z': 0, 'y': 0, 'x': 0}),
                (261, {'t': 1, 'z': 0, 'y': 0, 'x': 0}),
                (271, {'t': 2, 'z': 0, 'y': 0, 'x': 0}),
                (281, {'t': 3, 'z': 0, 'y': 0, 'x': 0}),
                (282, {'t': 3, 'z': 3, 'y': 0, 'x': 0}),
                (291, {'t': 4, 'z': 0, 'y': 0, 'x': 0}),
                (292, {'t': 4, 'z': 3, 'y': 0, 'x': 0}),
                (301, {'t': 5, 'z': 0, 'y': 0, 'x': 0}),
                (302, {'t': 5, 'z': -3, 'y': 0, 'x': 0}),
            ]
        gt_edges = [
            (261, 251),
            (271, 261),
            (281, 271),
            (282, 271),
            (291, 281),
            (292, 282),
            (301, 291),
            (302, 291),
            ]
        gt_track_graph = self.create_graph(gt_cells, gt_edges, roi)

        cells = [
                (2297, {'t': 0, 'z': 0, 'y': 0, 'x': 0}),
                (2788, {'t': 1, 'z': 0, 'y': 0, 'x': 0}),
                (2944, {'t': 2, 'z': 0, 'y': 0, 'x': 0}),
                (2453, {'t': 3, 'z': 0, 'y': 0, 'x': 0}),
                (3108, {'t': 2, 'z': 1, 'y': 0, 'x': 0}),
                (2990, {'t': 3, 'z': 3, 'y': 0, 'x': 0}),
                (2497, {'t': 4, 'z': 0, 'y': 0, 'x': 0}),
                (2498, {'t': 5, 'z': 0, 'y': 0, 'x': 0}),
                (2499, {'t': 5, 'z': -3, 'y': 0, 'x': 0}),
                (2991, {'t': 3, 'z': 4, 'y': 0, 'x': 0}),
                (2992, {'t': 4, 'z': 3, 'y': 0, 'x': 0}),
            ]
        edges = [
            (2788, 2297),
            (2944, 2788),
            (3108, 2788),
            (2990, 3108),
            (2453, 2944),
            (2497, 2453),
            (2498, 2497),
            (2499, 2497),
            (2992, 2991),
            ]
        for cell in cells:
            cell[1]['y'] += 1
        rec_track_graph = self.create_graph(cells, edges, roi)
        scores = e.evaluate(
            gt_track_graph, rec_track_graph, matching_threshold=2,
            sparse=True, validation_score=None, window_size=5,
            ignore_one_off_div_errors=False,
            fn_div_count_unconnected_parent=True)

        self.assertEqual(scores.identity_switches, 1)
        self.assertEqual(scores.gt_divisions, 2)
        self.assertEqual(scores.fn_divs_unconnected_child, 1)
        self.assertEqual(scores.fn_divs_unconnected_parent, 0)
        self.assertEqual(scores.fn_divs_no_connections, 0)
        self.assertEqual(scores.fn_divisions, 1)
        self.assertEqual(scores.fp_divisions, 1)
        self.assertEqual(scores.fn_edges, 0)
