import linajea.tracking
import logging
import linajea
import unittest
import daisy
import pymongo

logging.basicConfig(level=logging.INFO)
# logging.getLogger('linajea.tracking').setLevel(logging.DEBUG)


class TestSolver(unittest.TestCase):

    def delete_db(self, db_name, db_host):
        client = pymongo.MongoClient(db_host)
        client.drop_database(db_name)

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
            {'source': 1, 'target': 0, 'score': 1.0, 'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0, 'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0, 'prediction_distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0, 'prediction_distance': 0.0},
        ]
        db_name = 'linajea_test_solver'
        db_host = 'localhost'
        graph_provider = linajea.CandidateDatabase(
                db_name,
                db_host)
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        ps = {
                "cost_appear": 2.0,
                "cost_disappear": 2.0,
                "cost_split": 0,
                "weight_prediction_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 2.0,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
                "context": [2, 100, 100, 100],
            }
        job = {"num_workers": 5, "queue": "normal"}
        solve_config = linajea.config.SolveConfig(parameters=ps, job=job)

        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        linajea.tracking.nm_track(
                graph,
                solve_config,
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
        self.delete_db(db_name, db_host)

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
            {'source': 1, 'target': 0, 'score': 1.0, 'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0, 'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0, 'prediction_distance': 2.0},
        ]
        db_name = 'linajea_test_solver'
        db_host = 'localhost'
        graph_provider = linajea.CandidateDatabase(
                db_name,
                db_host)
        roi = daisy.Roi((0, 0, 0, 0), (5, 5, 5, 5))
        graph = graph_provider[roi]
        ps = {
                "cost_appear": 1.0,
                "cost_disappear": 1.0,
                "cost_split": 0,
                "weight_prediction_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 0.0,
                "max_cell_move": 1.0,
                "block_size": [5, 100, 100, 100],
                "context": [2, 100, 100, 100],
            }
        parameters = linajea.config.SolveParametersNonMinimalConfig(**ps)

        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        track_graph = linajea.tracking.TrackGraph(
                graph, frame_key='t', roi=graph.roi)
        solver = linajea.tracking.NMSolver(track_graph, parameters, 'selected')

        for node, data in track_graph.nodes(data=True):
            close = solver._check_node_close_to_roi_edge(node, data, 1)
            if node in [2, 4]:
                close = not close
            self.assertFalse(close)
        self.delete_db(db_name, db_host)
    

    def test_solver_multiple_configs(self):
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
            {'source': 1, 'target': 0, 'score': 1.0, 'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0, 'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0, 'prediction_distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0, 'prediction_distance': 0.0},
        ]
        db_name = 'linajea_test_solver'
        db_host = 'localhost'
        graph_provider = linajea.CandidateDatabase(
                db_name,
                db_host)
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        ps1 = {
                "cost_appear": 2.0,
                "cost_disappear": 2.0,
                "cost_split": 0,
                "weight_prediction_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 2.0,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
                "context": [2, 100, 100, 100],
            }
        ps2 = {
                "cost_appear": 2.0,
                "cost_disappear": 2.0,
                "cost_split":  10.0,
                "weight_prediction_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 2.0,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
                "context": [2, 100, 100, 100],
            }
        parameters = [ps1, ps2]
        keys = ['selected_1', 'selected_2']
        job = {"num_workers": 5, "queue": "normal"}
        solve_config = linajea.config.SolveConfig(parameters=parameters, job=job)

        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        linajea.tracking.nm_track(
                graph,
                solve_config,
                frame_key='t',
                selected_key=keys)

        selected_edges_1 = []
        selected_edges_2 = []
        for u, v, data in graph.edges(data=True):
            if data['selected_1']:
                selected_edges_1.append((u, v))
            if data['selected_2']:
                selected_edges_2.append((u, v))
        expected_result_1 = [
                (1, 0),
                (2, 1),
                (3, 1),
                (5, 3)
                ]
        expected_result_2 = [
                (1, 0),
                (3, 1),
                (5, 3)
                ]
        self.assertCountEqual(selected_edges_1, expected_result_1)
        self.assertCountEqual(selected_edges_2, expected_result_2)
        self.delete_db(db_name, db_host)
