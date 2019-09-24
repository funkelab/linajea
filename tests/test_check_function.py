from linajea.process_blockwise import check_function, write_done
from daisy import Block, Roi
import pymongo
import unittest


class TestCheckFunction(unittest.TestCase):
    def setUp(self):
        self.db_name = 'test_check_function'
        self.db_host = 'localhost'

    def tearDown(self):
        client = pymongo.MongoClient(self.db_host)
        client.drop_database(self.db_name)

    def test_check_function(self):
        step_name = 'predict'
        roi = Roi((0, 0, 0, 0), (10, 10, 10, 10))
        block = Block(roi, roi, roi)

        self.assertFalse(check_function(
            block, step_name, self.db_name, self.db_host))
        write_done(block, step_name, self.db_name, self.db_host)
        self.assertTrue(check_function(
            block, step_name, self.db_name, self.db_host))
