from linajea.gunpowder import WriteCells
import logging
import unittest
import numpy as np


try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class WriteCellsTestCase(unittest.TestCase):

    def get_parent_vectors(self):
        # zyx offset channels, t, z, y, x
        a = np.arange(3*3*3*3, dtype=np.float32).reshape((3, 1, 3, 3, 3))
        return a

    def test_get_avg_pv(self):
        parent_vectors = self.get_parent_vectors()
        print(parent_vectors)
        index = (0, 1, 1, 1)
        self.assertEqual(WriteCells.get_avg_pv(parent_vectors, index, 1),
                         (13., 40., 67.))
        self.assertEqual(WriteCells.get_avg_pv(parent_vectors, index, 3),
                         (13., 40., 67.))
        self.assertEqual(WriteCells.get_avg_pv(parent_vectors, index, 5),
                         (13., 40., 67.))
