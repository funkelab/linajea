import logging
import unittest

import networkx as nx

import daisy

import linajea.config
import linajea.tracking
import linajea.tracking.track
import linajea.tracking.cost_functions
import linajea.utils

logging.basicConfig(level=logging.INFO)


class TrackingConfig():
    def __init__(self, solve_config):
        self.solve = solve_config


class TestSolver(unittest.TestCase):

    def test_solver_basic(self):
        '''x
          3|         /-4
          2|        /--3---5
          1|   0---1
          0|        \\--2
            ------------------------------------ t
               0   1   2   3

        Should select 0, 1, 2, 3, 5
        '''

        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1, 'score': 2.0},
                {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1, 'score': 2.0},
                {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0, 'score': 2.0},
                {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2, 'score': 2.0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 3, 'score': 2.0},
                {'id': 5, 't': 3, 'z': 1, 'y': 1, 'x': 2, 'score': 2.0}
        ]

        edges = [
            {'source': 1, 'target': 0, 'score': 1.0,
             'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0,
             'prediction_distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0,
             'prediction_distance': 0.0},
        ]
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        ps = {
                "track_cost": 4.0,
                "weight_edge_score": 0.1,
                "weight_node_score": -0.1,
                "selection_constant": -1.0,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
            }
        job = {"num_workers": 5, "queue": "normal"}
        solve_config = linajea.config.SolveConfig(
            parameters=ps, job=job, context=[2, 100, 100, 100])
        solve_config.solver_type = "basic"
        config = TrackingConfig(solve_config)

        graph = nx.DiGraph()
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)

        linajea.tracking.track(
                graph,
                config,
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
            {'source': 1, 'target': 0, 'score': 1.0,
             'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0,
             'prediction_distance': 2.0},
        ]
        roi = daisy.Roi((0, 0, 0, 0), (5, 5, 5, 5))
        ps = {
                "track_cost": 4.0,
                "weight_edge_score": 0.1,
                "weight_node_score": -0.1,
                "selection_constant": -1.0,
                "max_cell_move": 1.0,
                "block_size": [5, 100, 100, 100],
            }
        parameters = linajea.config.SolveParametersConfig(**ps)

        graph = nx.DiGraph()
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)

        close_fn = linajea.tracking.cost_functions.is_close_to_roi_border(
            graph.roi, parameters.max_cell_move)
        for node, data in graph.nodes(data=True):
            close = close_fn(data)
            if node in [2, 4]:
                close = not close
            self.assertFalse(close)

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
            {'source': 1, 'target': 0, 'score': 1.0,
             'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0,
             'prediction_distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0,
             'prediction_distance': 0.0},
        ]
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        ps1 = {
                "track_cost": 4.0,
                "weight_edge_score": 0.1,
                "weight_node_score": -0.1,
                "selection_constant": -1.0,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
            }
        ps2 = {
                # Making all the values smaller increases the
                # relative cost of division
                "track_cost": 1.0,
                "weight_edge_score": 0.01,
                "weight_node_score": -0.01,
                "selection_constant": -0.1,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
            }
        parameters = [ps1, ps2]
        keys = ['selected_1', 'selected_2']
        job = {"num_workers": 5, "queue": "normal"}
        solve_config = linajea.config.SolveConfig(
            parameters=parameters, job=job, context=[2, 100, 100, 100])
        solve_config.solver_type = "basic"
        config = TrackingConfig(solve_config)

        graph = nx.DiGraph()
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)
        linajea.tracking.track(
                graph,
                config,
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

    def test_solver_cell_state(self):
        '''x
          3|         /-4
          2|        /--3---5
          1|   0---1
          0|        \\--2
            ------------------------------------ t
               0   1   2   3

        Should select 0, 1, 2, 3, 5
        '''

        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0,
                 'score_mother': 1,
                 "score_daughter": 0,
                 "score_continuation": 0},
                {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 1,
                 "score_continuation": 0},
                {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 1,
                 "score_continuation": 0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 5, 't': 3, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1}
        ]

        edges = [
            {'source': 1, 'target': 0, 'score': 1.0,
             'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0,
             'prediction_distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0,
             'prediction_distance': 0.0},
        ]
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        ps = {
                "track_cost": 4.0,
                "weight_edge_score": 0.1,
                "weight_node_score": -0.1,
                "selection_constant": -1.0,
                "weight_division": -0.1,
                "weight_child": -0.1,
                "weight_continuation": -0.1,
                "division_constant": 1,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
                "cell_state_key": "vgg_score",
            }
        job = {"num_workers": 5, "queue": "normal"}
        solve_config = linajea.config.SolveConfig(
            parameters=ps, job=job, context=[2, 100, 100, 100])
        solve_config.solver_type = "cell_state"
        config = TrackingConfig(solve_config)

        graph = nx.DiGraph()
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)
        linajea.tracking.track(
                graph,
                config,
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

    def test_solver_cell_state2(self):
        '''x
          3|         /-4
          2|        /--3---5
          1|   0---1
          0|        \\--2
            ------------------------------------ t
               0   1   2   3

        Should select 0, 1, 3, 5 due to vgg predicting continuation
        '''

        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
                {'id': 5, 't': 3, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0,
                 'score_mother': 0,
                 "score_daughter": 0,
                 "score_continuation": 1},
        ]

        edges = [
            {'source': 1, 'target': 0, 'score': 1.0,
             'prediction_distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0,
             'prediction_distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0,
             'prediction_distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0,
             'prediction_distance': 0.0},
        ]
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        ps = {
                "track_cost": 4.0,
                "weight_edge_score": 0.1,
                "weight_node_score": -0.1,
                "selection_constant": 0.0,
                "weight_division": -0.1,
                "weight_child": -0.1,
                "weight_continuation": -0.1,
                "division_constant": 1,
                "max_cell_move": 0.0,
                "block_size": [5, 100, 100, 100],
                "cell_state_key": "score"
            }
        job = {"num_workers": 5, "queue": "normal"}
        solve_config = linajea.config.SolveConfig(
            parameters=ps, job=job, context=[2, 100, 100, 100])
        solve_config.solver_type = "cell_state"
        config = TrackingConfig(solve_config)

        graph = nx.DiGraph()
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)
        linajea.tracking.track(
                graph,
                config,
                frame_key='t',
                selected_key='selected')

        selected_edges = []
        for u, v, data in graph.edges(data=True):
            if data['selected']:
                selected_edges.append((u, v))
        expected_result = [
                (1, 0),
                (3, 1),
                (5, 3)
                ]
        self.assertCountEqual(selected_edges, expected_result)
