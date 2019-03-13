import linajea.tracking
import logging
import linajea
import unittest
import daisy

logging.basicConfig(level=logging.INFO)
logging.getLogger('linajea.tracking').setLevel(logging.DEBUG)


class TestSolver(unittest.TestCase):

    def test_solver_basic(self):
        #   x
        #  3|         /-4
        #  2|        /--3---5
        #  1|   0---1
        #  0|        \--2
        #    ------------------------------------ t
        #       0   1   2   3

        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0},
                {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0},
                {'id': 5, 't': 3, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0}
        ]

        edges = [
            {'source': 1, 'target': 0, 'score': 1.0, 'distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0, 'distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0, 'distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0, 'distance': 0.0},
        ]
        graph_provider = linajea.CandidateDatabase(
                'linajea_test_solver',
                'localhost')
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        parameters = linajea.tracking.TrackingParameters()
        parameters.cost_appear = 2.0
        parameters.cost_disappear = 2.0
        parameters.cost_split = 0
        parameters.weight_distance_cost = 0.1
        parameters.weight_node_score = 1.0
        parameters.threshold_node_score = 0.0
        parameters.threshold_edge_score = 0.0
        parameters.max_cell_move = 1.0

        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        linajea.tracking.track(
                graph,
                parameters,
                frame_key='t',
                selected_key='selected')

        selected_edges = []
        for u, v, data in graph.edges(data=True):
            if data['selected']:
                selected_edges.append((u, v))
        expected_result = [
                (1, 0),
                (2, 1),
                (3, 1),
                (5, 3)
                ]
        self.assertCountEqual(selected_edges, expected_result)

    def test_solver_node_close_to_edge(self):
        #   x
        #  3|         /-4
        #  2|        /--3
        #  1|   0---1
        #  0|        \--2
        #    ------------------------------------ t
        #       0   1   2

        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0},
                {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 4,  'score': 2.0}
        ]

        edges = [
            {'source': 1, 'target': 0, 'score': 1.0, 'distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0, 'distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0, 'distance': 2.0},
        ]
        graph_provider = linajea.CandidateDatabase(
                'linajea_test_solver',
                'localhost')
        roi = daisy.Roi((0, 0, 0, 0), (5, 5, 5, 5))
        graph = graph_provider[roi]
        parameters = linajea.tracking.TrackingParameters()
        parameters.cost_appear = 1.0
        parameters.cost_disappear = 1.0
        parameters.cost_split = 0
        parameters.weight_distance_cost = 0.1
        parameters.weight_node_score = 1.0
        parameters.threshold_node_score = 0.0
        parameters.threshold_edge_score = 0.0

        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        track_graph = linajea.tracking.TrackGraph(
                graph, frame_key='t', roi=graph.roi)
        solver = linajea.tracking.Solver(track_graph, parameters, 'selected')

        for node, data in track_graph.nodes(data=True):
            close = solver._check_node_close_to_roi_edge(node, data, 1)
            if node in [2, 4]:
                close = not close
            self.assertFalse(close)
