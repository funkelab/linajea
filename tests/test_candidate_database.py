from linajea import CandidateDatabase
import linajea.tracking
from daisy import Roi
from unittest import TestCase
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatabaseTestCase(TestCase):
    def test_database_creation(self):
        db_name = 'test_linajea_database'
        mongo_url = 'localhost'
        total_roi = Roi((0, 0, 0, 0), (10, 100, 100, 100))

        candidate_db = CandidateDatabase(
                db_name,
                mongo_url,
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

        print("Creating new database to read data")
        compare_db = CandidateDatabase(
                db_name,
                mongo_url,
                mode='r',
                total_roi=total_roi)

        compare_sub_graph = compare_db[total_roi]

        point_ids = [p[0] for p in points]
        self.assertCountEqual(compare_sub_graph.nodes, point_ids)
        self.assertCountEqual(compare_sub_graph.edges, edges)

    def test_read_existing_database(self):
        db_name = 'linajea_setup02_test'
        mongo_url = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.'\
                    'int.janelia.org:27023/admin?replicaSet=rsFunke'
        candidate_db = CandidateDatabase(
                db_name,
                mongo_url,
                mode='r')
        roi = Roi((200, 1000, 1000, 1000), (3, 1000, 1000, 1000))
        subgraph = candidate_db[roi]
        logger.info("Number of nodes: {}".format(len(subgraph.nodes)))
        logger.info("Number of edges: {}".format(len(subgraph.edges)))


class TestParameterIds(TestCase):
    def test_unique_id_one_worker(self):
        db = CandidateDatabase(
                'test_linajea_db',
                'localhost',
                mode='w')
        db._MongoDbGraphProvider__connect()
        db._MongoDbGraphProvider__open_db()
        db.database['parameters'].drop()
        ps = {
                "cost_appear": 1.0,
                "cost_disappear": 1.0,
                "cost_split": 0,
                "weight_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 0.0,
                "max_cell_move": 1.0,
            }
        block_size = [5, 500, 500, 500]
        context = [2, 100, 100, 100]
        for i in range(10):
            tp = linajea.tracking.TrackingParameters(**ps)
            tp.cost_appear = i
            _id = db.get_parameters_id(tp, block_size, context)
            self.assertEqual(_id, i + 1)

    def test_unique_id_multi_worker(self):
        db = linajea.CandidateDatabase(
                'test_linajea_db_multi_worker',
                'localhost',
                mode='w')
        db._MongoDbGraphProvider__connect()
        db._MongoDbGraphProvider__open_db()
        db.database['parameters'].drop()
        tps = []
        ps = {
                "cost_appear": 1.0,
                "cost_disappear": 1.0,
                "cost_split": 0,
                "weight_distance_cost": 0.1,
                "weight_node_score": 1.0,
                "threshold_node_score": 0.0,
                "threshold_edge_score": 0.0,
                "max_cell_move": 1.0,
            }
        block_size = [5, 500, 500, 500]
        context = [2, 100, 100, 100]
        for i in range(10):
            tp = linajea.tracking.TrackingParameters(**ps)
            tp.cost_appear = i
            tps.append(tp)

        class ID_Process(mp.Process):
            def __init__(self, db, parameters):
                super(ID_Process, self).__init__()
                self.db = db
                self.params = parameters

            def run(self):
                return self.db.get_parameters_id(self.params, block_size, context)

        processes = [ID_Process(db, tp) for tp in tps]
        for _id, process in enumerate(processes):
            process.start()
