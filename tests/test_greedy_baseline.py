import logging
import pymongo
import unittest

import daisy

import linajea.utils
import linajea.tracking

logging.basicConfig(level=logging.INFO)


class TestGreedy(unittest.TestCase):

    def setUp(self):
        self.db_host = 'localhost'
        try:
            client = pymongo.MongoClient(
                self.db_host, serverSelectionTimeoutMS=0)
            client.list_databases()
        except pymongo.errors.ServerSelectionTimeoutError:
            self.skipTest("No MongoDB server found")

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)

    def test_greedy_basic_db(self):
        #   x
        #  3|   1---2 \
        #  2|     /    -5 (x=2.1)
        #  1|   0---3 /
        #  0|        \--4
        #    ------------------------------------ t
        #       0   1   2   3
        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 1, 't': 0, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0},
                {'id': 2, 't': 1, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0},
                {'id': 3, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0},
                {'id': 5, 't': 2, 'z': 1, 'y': 1, 'x': 2.1,  'score': 2.0}
        ]

        edges = [
            {'source': 3, 'target': 0, 'score': 1.0, 'distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'distance': 0.0},
            {'source': 5, 'target': 2, 'score': 1.0, 'distance': 0.9},
            {'source': 4, 'target': 3, 'score': 1.0, 'distance': 1.0},
            {'source': 5, 'target': 3, 'score': 1.0, 'distance': 1.1},
        ]
        self.db_name = 'linajea_test_solver'
        self.db_host = 'localhost'
        graph_provider = linajea.utils.CandidateDatabase(
            self.db_name,
            self.db_host,
            mode='w')
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        graph.write_nodes()
        graph.write_edges()

        linajea.tracking.greedy_track(
            db_name=self.db_name,
            db_host=self.db_host,
            selected_key='selected_1',
            metric='distance',
            frame_key='t')

        read_db = linajea.utils.CandidateDatabase(
            self.db_name,
            self.db_host,
            mode='r',
            parameters_id=1)
        selected_graph = read_db.get_selected_graph(roi)

        selected_edges = []
        for u, v, data in selected_graph.edges(data=True):
            if data['selected_1']:
                selected_edges.append((u, v))
        expected_result = [
                (3, 0),
                (2, 1),
                (4, 3),
                (5, 2)
                ]
        self.assertCountEqual(selected_edges, expected_result)

    def test_greedy_basic_graph(self):
        #   x
        #  3|   1---2 \
        #  2|     /    -5 (x=2.1)
        #  1|   0---3 /
        #  0|        \--4
        #    ------------------------------------ t
        #       0   1   2   3
        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 1, 't': 0, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0},
                {'id': 2, 't': 1, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0},
                {'id': 3, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0},
                {'id': 5, 't': 2, 'z': 1, 'y': 1, 'x': 2.1,  'score': 2.0}
        ]

        edges = [
            {'source': 3, 'target': 0, 'score': 1.0, 'distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'distance': 0.0},
            {'source': 5, 'target': 2, 'score': 1.0, 'distance': 0.9},
            {'source': 4, 'target': 3, 'score': 1.0, 'distance': 1.0},
            {'source': 5, 'target': 3, 'score': 1.0, 'distance': 1.1},
        ]
        self.db_name = 'linajea_test_solver'
        self.db_host = 'localhost'
        graph_provider = linajea.utils.CandidateDatabase(
            self.db_name,
            self.db_host,
            mode='w')
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        linajea.tracking.greedy_track(
                graph=graph,
                selected_key='selected',
                metric='distance',
                frame_key='t')
        selected_edges = []
        for u, v, data in graph.edges(data=True):
            if data['selected']:
                selected_edges.append((u, v))
        expected_result = [
                (3, 0),
                (2, 1),
                (4, 3),
                (5, 2)
                ]
        self.assertCountEqual(selected_edges, expected_result)

    def test_greedy_split(self):
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
        self.db_name = 'linajea_test_solver'
        self.db_host = 'localhost'
        graph_provider = linajea.utils.CandidateDatabase(
            self.db_name,
            self.db_host,
            mode='w')
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        linajea.tracking.greedy_track(
                graph=graph,
                selected_key='selected',
                metric='distance',
                frame_key='t')
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

    def test_greedy_node_threshold(self):
        #   x
        #  3|         /-4 \
        #  2|        /--3---5
        #  1|   0---1
        #  0|        \--2
        #    ------------------------------------ t
        #       0   1   2   3
        cells = [
                {'id': 0, 't': 0, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 1, 't': 1, 'z': 1, 'y': 1, 'x': 1,  'score': 2.0},
                {'id': 2, 't': 2, 'z': 1, 'y': 1, 'x': 0,  'score': 2.0},
                {'id': 3, 't': 2, 'z': 1, 'y': 1, 'x': 2,  'score': 1.0},
                {'id': 4, 't': 2, 'z': 1, 'y': 1, 'x': 3,  'score': 2.0},
                {'id': 5, 't': 3, 'z': 1, 'y': 1, 'x': 2,  'score': 2.0}
        ]

        edges = [
            {'source': 1, 'target': 0, 'score': 1.0, 'distance': 0.0},
            {'source': 2, 'target': 1, 'score': 1.0, 'distance': 1.0},
            {'source': 3, 'target': 1, 'score': 1.0, 'distance': 1.0},
            {'source': 4, 'target': 1, 'score': 1.0, 'distance': 2.0},
            {'source': 5, 'target': 3, 'score': 1.0, 'distance': 0.0},
            {'source': 5, 'target': 4, 'score': 1.0, 'distance': 1.0},
        ]
        self.db_name = 'linajea_test_solver'
        self.db_host = 'localhost'
        graph_provider = linajea.utils.CandidateDatabase(
            self.db_name,
            self.db_host,
            mode='w')
        roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
        graph = graph_provider[roi]
        graph.add_nodes_from([(cell['id'], cell) for cell in cells])
        graph.add_edges_from([(edge['source'], edge['target'], edge)
                              for edge in edges])
        linajea.tracking.greedy_track(
                graph=graph,
                selected_key='selected',
                metric='distance',
                frame_key='t',
                node_threshold=1.5)
        selected_edges = []
        for u, v, data in graph.edges(data=True):
            if data['selected']:
                selected_edges.append((u, v))
        expected_result = [
                (1, 0),
                (4, 1),
                (5, 4)
                ]
        self.assertCountEqual(selected_edges, expected_result)
