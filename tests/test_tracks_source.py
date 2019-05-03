from linajea.gunpowder import TracksSource, AddParentVectors
import logging
import os
import gunpowder as gp
import unittest
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

TEST_FILE = 'testdata.txt'


class TracksSourceTestCase(unittest.TestCase):

    def setUp(self):
        # t z y x id parent_id track_id
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

    def tearDown(self):
        os.remove(TEST_FILE)

    def test_parent_location(self):
        points = gp.PointsKey("POINTS")
        ts = TracksSource(
                TEST_FILE,
                points)

        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((5, 5, 5, 5)))

        ts.setup()
        b = ts.provide(request)
        points = b[points].data
        self.assertListEqual([0.0, 0.0, 0.0, 0.0],
                             list(points[1].location))
        self.assertListEqual([1.0, 0.0, 0.0, 0.0],
                             list(points[2].location))
        self.assertListEqual([1.0, 1.0, 2.0, 3.0],
                             list(points[3].location))
        self.assertListEqual([2.0, 2.0, 2.0, 2.0],
                             list(points[4].location))

    def test_delete_points_in_context(self):
        points = gp.PointsKey("POINTS")
        pv_array = gp.ArrayKey("PARENT_VECTORS")
        mask = gp.ArrayKey("MASK")
        radius = [0.1, 0.1, 0.1, 0.1]
        ts = TracksSource(
                TEST_FILE,
                points)
        apv = AddParentVectors(
                points,
                pv_array,
                mask,
                radius)
        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((1, 4, 4, 4)))
        request.add(
                pv_array,
                gp.Coordinate((1, 4, 4, 4)))
        request.add(
                mask,
                gp.Coordinate((1, 4, 4, 4)))

        pipeline = (
                ts +
                gp.Pad(points, None) +
                apv)
        with gp.build(pipeline):
            pipeline.request_batch(request)

    def test_add_parent_vectors(self):
        points = gp.PointsKey("POINTS")
        pv_array = gp.ArrayKey("PARENT_VECTORS")
        mask = gp.ArrayKey("MASK")
        radius = [0.1, 0.1, 0.1, 0.1]
        ts = TracksSource(
                TEST_FILE,
                points)
        apv = AddParentVectors(
                points,
                pv_array,
                mask,
                radius)
        request = gp.BatchRequest()
        request.add(
                points,
                gp.Coordinate((3, 4, 4, 4)))
        request.add(
                pv_array,
                gp.Coordinate((1, 4, 4, 4)))
        request.add(
                mask,
                gp.Coordinate((1, 4, 4, 4)))

        pipeline = (
                ts +
                gp.Pad(points, None) +
                apv)
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        points = batch[points].data
        expected_mask = np.zeros(shape=(1, 4, 4, 4))
        expected_mask[0, 0, 0, 0] = 1
        expected_mask[0, 1, 2, 3] = 1

        expected_parent_vectors_z = np.zeros(shape=(1, 4, 4, 4))
        expected_parent_vectors_z[0, 1, 2, 3] = -1.0

        expected_parent_vectors_y = np.zeros(shape=(1, 4, 4, 4))
        expected_parent_vectors_y[0, 1, 2, 3] = -2.0

        expected_parent_vectors_x = np.zeros(shape=(1, 4, 4, 4))
        expected_parent_vectors_x[0, 1, 2, 3] = -3.0
        # print("MASK")
        # print(batch[mask].data)
        self.assertListEqual(expected_mask.tolist(), batch[mask].data.tolist())

        parent_vectors = batch[pv_array].data
        self.assertListEqual(expected_parent_vectors_z.tolist(),
                             parent_vectors[0].tolist())
        self.assertListEqual(expected_parent_vectors_y.tolist(),
                             parent_vectors[1].tolist())
        self.assertListEqual(expected_parent_vectors_x.tolist(),
                             parent_vectors[2].tolist())
