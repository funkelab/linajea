import daisy
import numpy as np
import logging
import unittest
from linajea import downsample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)


class DownsampleTestCase(unittest.TestCase):
    # Downsampling by 1s does nothing
    def test_downsample_1(self):
        parent_vectors_data = np.random.rand(3, 5, 10, 10, 10)
        parent_vectors = daisy.Array(parent_vectors_data,
                                    daisy.Roi((0,0,0,0), (5, 10, 10, 10)),
                                    voxel_size=(1,1,1,1))
        downsample_factors = [1,1,1]
        result = downsample(parent_vectors, downsample_factors)
        self.assertTrue(np.array_equal(parent_vectors_data, result.data))
    
    # Downsampling changes shape and voxel size as expected
    def test_downsample_2(self):
        parent_vectors_data = np.zeros((3, 5, 10, 10, 10))
        parent_vectors = daisy.Array(parent_vectors_data,
                                    daisy.Roi((0,0,0,0), (5, 20, 10, 10)),
                                    voxel_size=(1,2,1,1))
        downsample_factors = [1,2,2]
        
        expected_data = np.zeros((3, 5, 10, 5, 5))
        expected = daisy.Array(expected_data,
                                daisy.Roi((0,0,0,0), (5,20,10,10)),
                                voxel_size=(1,2,2,2))
        result = downsample(parent_vectors, downsample_factors)
        self.assertTrue(np.array_equal(expected.data, result.data))
        self.assertEqual(result.voxel_size, expected.voxel_size)

    # Downsampling changes values as expected (one time step)
    def test_downsample_3(self):
        parent_vectors_data = np.array([[1, 2, 3],
                               [1, 2, 3],
                               [4, 5, 6]])
        parent_vectors_data = parent_vectors_data.reshape(3,1,3,1,1)
        logger.debug("Parent vectors: {}".format(parent_vectors_data))
        parent_vectors = daisy.Array(parent_vectors_data,
                                    daisy.Roi((0,0,0,0), (1, 3, 3, 3)),
                                    voxel_size=(1,1,3,3))
        downsample_factors = [3,1,1]
        
        expected_data = np.array([2,2,5]).reshape(3,1,1,1,1)
        logger.debug("Expected {}".format(expected_data))
        expected = daisy.Array(expected_data,
                                daisy.Roi((0,0,0,0), (1,3,3,3)),
                                voxel_size=(1,3,3,3))
        result = downsample(parent_vectors, downsample_factors)
        logger.debug("Result data: {}".format(result.data))
        self.assertTrue(np.array_equal(expected.data, result.data))
        self.assertEqual(result.voxel_size, expected.voxel_size)
    
    # Downsampling changes values as expected (multiple time steps)
    def test_downsample_4(self):
        parent_vectors_data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7])
        parent_vectors_data = parent_vectors_data.reshape(3,2,3,1,1)
        logger.debug("Parent vectors: {}".format(parent_vectors_data))
        parent_vectors = daisy.Array(parent_vectors_data,
                                    daisy.Roi((0,0,0,0), (2, 3, 3, 3)),
                                    voxel_size=(1,1,3,3))
        downsample_factors = [3,1,1]
        
        expected_data = np.array([1, 2, 3, 4, 5, 6]).reshape(3,2,1,1,1)
        logger.debug("Expected {}".format(expected_data))
        expected = daisy.Array(expected_data,
                                daisy.Roi((0,0,0,0), (2,3,3,3)),
                                voxel_size=(1,3,3,3))
        result = downsample(parent_vectors, downsample_factors)
        logger.debug("Result data: {}".format(result.data))
        self.assertTrue(np.array_equal(expected.data, result.data))
        self.assertEqual(result.voxel_size, expected.voxel_size)


