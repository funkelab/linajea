import logging
import os
import unittest

import numpy as np

import gunpowder as gp

from linajea.gunpowder_nodes import TracksSource, AddMovementVectors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('linajea').setLevel(logging.DEBUG)


TEST_FILE = 'testdata.txt'
TEST_FILE_WITH_HEADER = 'testdata_with_header.txt'


class TracksSourceTestCase(unittest.TestCase):

    def setUp(self):
        h = 't z y x cell_id parent_id track_id\n'
        p1 = "0.0 0.0 0.0 0.0 1 -1 0\n"
        p2 = "1.0 0.0 0.0 0.0 2 1 0\n"
        p3 = "1.0 1.0 2.0 3.0 3 1 0\n"
        p4 = "2.0 2.0 2.0 2.0 4 3 0\n"
        p5 = "1.0 1.0 1.0 1.0 5 -1 0\n"
        with open(TEST_FILE, 'w') as f:
            f.write(p1)
            f.write(p2)
            f.write(p3)
            f.write(p4)
            f.write(p5)
        with open(TEST_FILE_WITH_HEADER, 'w') as f:
            f.write(h)
            f.write(p1)
            f.write(p2)
            f.write(p3)
            f.write(p4)
            f.write(p5)

    def tearDown(self):
        os.remove(TEST_FILE)
        os.remove(TEST_FILE_WITH_HEADER)

    def test_parent_location(self):
        points = gp.GraphKey("POINTS")
        ts = TracksSource(
                TEST_FILE,
                points)

        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((5, 5, 5, 5)))

        ts.setup()
        b = ts.provide(request)
        points = [n.location for n in b[points].nodes]
        self.assertListEqual([0.0, 0.0, 0.0, 0.0],
                             list(points[0]))
        self.assertListEqual([1.0, 0.0, 0.0, 0.0],
                             list(points[1]))
        self.assertListEqual([1.0, 1.0, 2.0, 3.0],
                             list(points[2]))
        self.assertListEqual([2.0, 2.0, 2.0, 2.0],
                             list(points[3]))

    def test_csv_header(self):
        points = gp.GraphKey("POINTS")
        tswh = TracksSource(
                TEST_FILE_WITH_HEADER,
                points)

        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((5, 5, 5, 5)))

        tswh.setup()
        b = tswh.provide(request)
        points = [n.location for n in b[points].nodes]
        self.assertListEqual([0.0, 0.0, 0.0, 0.0],
                             list(points[0]))
        self.assertListEqual([1.0, 0.0, 0.0, 0.0],
                             list(points[1]))
        self.assertListEqual([1.0, 1.0, 2.0, 3.0],
                             list(points[2]))
        self.assertListEqual([2.0, 2.0, 2.0, 2.0],
                             list(points[3]))

    def test_delete_points_in_context(self):
        points = gp.GraphKey("POINTS")
        mv_array = gp.ArrayKey("MOVEMENT_VECTORS")
        mask = gp.ArrayKey("MASK")
        radius = [0.1, 0.1, 0.1, 0.1]
        ts = TracksSource(
                TEST_FILE,
                points)
        amv = AddMovementVectors(
                points,
                mv_array,
                mask,
                radius)
        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((1, 4, 4, 4)))
        request.add(
                mv_array,
                gp.Coordinate((1, 4, 4, 4)))
        request.add(
                mask,
                gp.Coordinate((1, 4, 4, 4)))

        pipeline = (
                ts +
                gp.Pad(points, None) +
                amv)
        with gp.build(pipeline):
            pipeline.request_batch(request)

    def test_add_movement_vectors(self):
        points = gp.GraphKey("POINTS")
        mv_array = gp.ArrayKey("MOVEMENT_VECTORS")
        mask = gp.ArrayKey("MASK")
        radius = [0.1, 0.1, 0.1, 0.1]
        ts = TracksSource(
                TEST_FILE,
                points)
        amv = AddMovementVectors(
                points,
                mv_array,
                mask,
                radius)
        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((3, 4, 4, 4)))
        request.add(
                mv_array,
                gp.Coordinate((1, 4, 4, 4)))
        request.add(
                mask,
                gp.Coordinate((1, 4, 4, 4)))

        pipeline = (
                ts +
                gp.Pad(points, None) +
                amv)
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        points = [n.location for n in batch[points].nodes]
        expected_mask = np.zeros(shape=(1, 4, 4, 4))
        expected_mask[0, 0, 0, 0] = 1
        expected_mask[0, 1, 2, 3] = 1

        expected_movement_vectors_z = np.zeros(shape=(1, 4, 4, 4))
        expected_movement_vectors_z[0, 1, 2, 3] = -1.0

        expected_movement_vectors_y = np.zeros(shape=(1, 4, 4, 4))
        expected_movement_vectors_y[0, 1, 2, 3] = -2.0

        expected_movement_vectors_x = np.zeros(shape=(1, 4, 4, 4))
        expected_movement_vectors_x[0, 1, 2, 3] = -3.0
        # print("MASK")
        # print(batch[mask].data)
        self.assertListEqual(expected_mask.tolist(), batch[mask].data.tolist())

        movement_vectors = batch[mv_array].data
        self.assertListEqual(expected_movement_vectors_z.tolist(),
                             movement_vectors[0].tolist())
        self.assertListEqual(expected_movement_vectors_y.tolist(),
                             movement_vectors[1].tolist())
        self.assertListEqual(expected_movement_vectors_x.tolist(),
                             movement_vectors[2].tolist())
