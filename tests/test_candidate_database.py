from linajea import CandidateDatabase
import linajea.tracking
from linajea.evaluation import Report
from daisy import Roi
from unittest import TestCase
import logging
import multiprocessing as mp
import pymongo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatabaseTestCase(TestCase):

    def delete_db(self, db_name, db_host):
        client = pymongo.MongoClient(db_host)
        client.drop_database(db_name)

    def test_database_creation(self):
        db_name = 'test_linajea_database'
        db_host = 'localhost'
        total_roi = Roi((0, 0, 0, 0), (10, 100, 100, 100))

        candidate_db = CandidateDatabase(
                db_name,
                db_host,
                mode='w',
                total_roi=total_roi)

        sub_graph = candidate_db[total_roi]
        points = []
        for i in range(5):
            points.append((i, {
                'id': i,
                't': i,
                'z': i,
                'y': i,
                'x': i,
                }))
        edges = []
        for i in range(4):
            edges.append((i + 1, i))
        sub_graph.add_nodes_from(points)
        sub_graph.add_edges_from(edges)
        sub_graph.write_nodes()
        sub_graph.write_edges()

        logger.debug("Creating new database to read data")
        compare_db = CandidateDatabase(
                db_name,
                db_host,
                mode='r',
                total_roi=total_roi)

        compare_sub_graph = compare_db[total_roi]

        point_ids = [p[0] for p in points]
        self.assertCountEqual(compare_sub_graph.nodes, point_ids)
        self.assertCountEqual(compare_sub_graph.edges, edges)
        self.delete_db(db_name, db_host)

    def test_read_existing_database(self):
        db_name = 'linajea_setup08_400000'
        mongo_url = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.'\
                    'int.janelia.org:27023/admin?replicaSet=rsFunke'
        candidate_db = CandidateDatabase(
                db_name,
                mongo_url,
                mode='r')
        roi = Roi((200, 1000, 1000, 1000), (3, 1000, 1000, 1000))
        subgraph = candidate_db[roi]
        self.assertTrue(subgraph.number_of_nodes() > 0)
        self.assertTrue(subgraph.number_of_edges() > 0)

    def test_get_selected_graph_and_reset_selection(self):
        db_name = 'test_linajea_database'
        db_host = 'localhost'
        total_roi = Roi((0, 0, 0, 0), (5, 10, 10, 10))

        write_db = CandidateDatabase(
                db_name,
                db_host,
                mode='w',
                total_roi=total_roi)

        sub_graph = write_db[total_roi]
        points = [
                (1, {'t': 0, 'z': 1, 'y': 2, 'x': 3}),
                (2, {'t': 1, 'z': 1, 'y': 2, 'x': 3}),
                (3, {'t': 2, 'z': 1, 'y': 2, 'x': 3}),
                (4, {'t': 3, 'z': 1, 'y': 2, 'x': 3}),
                (5, {'t': 2, 'z': 5, 'y': 2, 'x': 3}),
                (6, {'t': 3, 'z': 5, 'y': 2, 'x': 3}),
                ]
        edges = [
                (2, 1, {'selected_1': True}),
                (3, 2, {'selected_1': True}),
                (4, 3, {'selected_1': True}),
                (5, 2, {'selected_1': False}),
                (6, 5, {'selected_1': False}),
                ]
        sub_graph.add_nodes_from(points)
        sub_graph.add_edges_from(edges)
        sub_graph.write_nodes()
        sub_graph.write_edges()

        logger.debug("Creating new database to read data")
        read_db = CandidateDatabase(
                db_name,
                db_host,
                mode='r',
                parameters_id=1)
        selected_graph = read_db.get_selected_graph(total_roi)
        self.assertEqual(selected_graph.number_of_nodes(), 4)
        self.assertEqual(selected_graph.number_of_edges(), 3)

        read_db.reset_selection()
        unselected_graph = read_db.get_selected_graph(total_roi)
        self.assertEqual(unselected_graph.number_of_nodes(), 0)
        self.assertEqual(unselected_graph.number_of_edges(), 0)

    def test_write_and_get_score(self):
        db_name = 'test_linajea_database'
        db_host = 'localhost'
        ps = {
                "cost_appear": 2.0,
                "cost_disappear": 2.0,
                "cost_split": 0,
                "weight_prediction_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 0.0,
                "max_cell_move": 1.0,
                "block_size": [5, 100, 100, 100],
                "context": [2, 100, 100, 100],
            }
        parameters = linajea.tracking.TrackingParameters(**ps)

        db = CandidateDatabase(
                db_name,
                db_host)
        params_id = db.get_parameters_id(parameters)

        score = Report()
        score.gt_edges = 2
        score.matched_edges = 2
        score.fp_edges = 1
        score.fn_edges = 0
        db.write_score(params_id, score)
        score_dict = db.get_score(params_id)

        compare_dict = score.__dict__
        compare_dict.update(db.get_parameters(params_id))
        self.assertDictEqual(compare_dict, score_dict)


class TestParameterIds(TestCase):

    def delete_db(self, db_name, db_host):
        client = pymongo.MongoClient(db_host)
        client.drop_database(db_name)

    def get_tracking_params(self):
        return {
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

    def test_unique_id_one_worker(self):
        db_name = 'test_linajea_db'
        db_host = 'localhost'
        db = CandidateDatabase(
                db_name,
                db_host,
                mode='w')
        for i in range(10):
            tp = linajea.tracking.TrackingParameters(
                    **self.get_tracking_params())
            tp.cost_appear = i
            _id = db.get_parameters_id(tp)
            self.assertEqual(_id, i + 1)
        self.delete_db(db_name, db_host)

    def test_unique_id_multi_worker(self):
        db_name = 'test_linajea_db_multi_worker'
        db_host = 'localhost'
        db = linajea.CandidateDatabase(
                db_name,
                db_host,
                mode='w')
        tps = []
        for i in range(10):
            tp = linajea.tracking.TrackingParameters(
                    **self.get_tracking_params())
            tp.cost_appear = i
            tps.append(tp)

        class ID_Process(mp.Process):
            def __init__(self, db, parameters):
                super(ID_Process, self).__init__()
                self.db = db
                self.params = parameters

            def run(self):
                return self.db.get_parameters_id(self.params)

        processes = [ID_Process(db, tp) for tp in tps]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        self.delete_db(db_name, db_host)
