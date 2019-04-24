from linajea.evaluation.match import match, get_edge_costs, match_edges
import unittest
import networkx as nx
from linajea.tracking import TrackGraph


class TestEvalMatch(unittest.TestCase):

    def test_match_simple(self):
        costs = {(1, 1): 10}
        no_match_cost = 20

        matches, cost = match(costs, no_match_cost)
        expected_matches = [(1, 1)]
        self.assertCountEqual(matches, expected_matches)
        self.assertEqual(cost, 10)

        no_match_cost = 3

        matches, cost = match(costs, no_match_cost)
        expected_matches = []
        self.assertCountEqual(matches, expected_matches)
        self.assertEqual(cost, 6)

        costs = {
                (1, 1): 10,
                (1, 2): 15,
                }
        no_match_cost = 20

        matches, cost = match(costs, no_match_cost)
        expected_matches = [(1, 1)]
        self.assertCountEqual(matches, expected_matches)
        self.assertEqual(cost, 30)

    def test_default_cost(self):

        costs = {
                (1, 1): 10,
                (2, 2): 10,
                (1, 3): 12,
                (3, 2): 14,
                }
        no_match_cost = 20

        matches, cost = match(costs, no_match_cost)
        expected_matches = [(1, 1), (2, 2)]
        self.assertCountEqual(matches, expected_matches)
        self.assertEqual(cost, 60)

        no_match_cost = 15

        matches, cost = match(costs, no_match_cost)
        expected_matches = [(1, 1), (2, 2)]
        self.assertCountEqual(matches, expected_matches)
        self.assertEqual(cost, 50)

    def test_get_edge_costs(self):
        edges_x = [
                (1, 2),
                ]
        edges_y = [
                (1, 2),
                (3, 4),
                ]
        node_pairs_xy = {
                1: [(1, 0.5), (3, 0.5)],
                2: [(2, 0.5)]
                }
        edge_costs = get_edge_costs(edges_x, edges_y, node_pairs_xy)
        expected_edge_costs = {
                (0, 0): 1.0
                }
        self.assertEqual(edge_costs, expected_edge_costs)

    def test_match_tracks_simple(self):
        cells_x = [
                (0, {'t': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0}),
                (1, {'t': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0}),
        ]
        edges_x = [
            (1, 0, {'score': 1.0, 'distance': 0.0}),
        ]

        cells_y = [
                (0, {'t': 0, 'z': 0, 'y': 1, 'x': 1,  'score': 2.0}),
                (1, {'t': 1, 'z': 0, 'y': 1, 'x': 1,  'score': 2.0}),
                (2, {'t': 0, 'z': 1, 'y': 0, 'x': 1,  'score': 2.0}),
                (3, {'t': 1, 'z': 1, 'y': -2, 'x': 1,  'score': 2.0}),
        ]
        edges_y = [
            (1, 0, {'score': 1.0, 'distance': 0.0}),
            (3, 2, {'score': 1.0, 'distance': 3.0}),
        ]

        graph_x = nx.DiGraph()
        graph_x.add_nodes_from(cells_x)
        graph_x.add_edges_from(edges_x)
        track_graph_x = TrackGraph(graph_x)

        graph_y = nx.DiGraph()
        graph_y.add_nodes_from(cells_y)
        graph_y.add_edges_from(edges_y)
        track_graph_y = TrackGraph(graph_y)
        ex, ey, matches = match_edges(track_graph_x, track_graph_y, 2)
        edge_matches = [(ex[x_ind], ey[y_ind]) for x_ind, y_ind in matches]
        expected_edge_matches = [
                ((1, 0), (1, 0)),
                ]
        self.assertEqual(edge_matches, expected_edge_matches)
