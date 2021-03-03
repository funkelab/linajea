from funlib import math
import gunpowder as gp
import numpy as np
import pymongo
from pymongo.errors import BulkWriteError
import logging

logger = logging.getLogger(__name__)


class WriteCells(gp.BatchFilter):

    def __init__(
            self,
            maxima,
            cell_indicator,
            parent_vectors,
            score_threshold,
            db_host,
            db_name,
            edge_length=1,
            volume_shape=None):
        '''Edge length indicates the length of the edge of the cube
        from which parent vectors will be read. The cube will be centered
        around the maxima, and predictions within the cube of voxels
        will be averaged to get the parent vector to store in the db
        '''

        self.maxima = maxima
        self.cell_indicator = cell_indicator
        self.parent_vectors = parent_vectors
        self.score_threshold = score_threshold
        self.db_host = db_host
        self.db_name = db_name
        self.client = None
        assert edge_length % 2 == 1, "Edge length should be odd"
        self.edge_length = edge_length
        self.volume_shape = volume_shape

    def process(self, batch, request):

        if self.client is None:
            self.client = pymongo.MongoClient(host=self.db_host)
            self.db = self.client[self.db_name]
            create_indices = 'nodes' not in self.db.list_collection_names()
            self.cells = self.db['nodes']
            if create_indices:
                self.cells.create_index(
                    [
                        (loc, pymongo.ASCENDING)
                        for loc in ['t', 'z', 'y', 'x']
                    ],
                    name='position')
                self.cells.create_index(
                    [
                        ('id', pymongo.ASCENDING)
                    ],
                    name='id',
                    unique=True)

        roi = batch[self.maxima].spec.roi
        voxel_size = batch[self.maxima].spec.voxel_size

        maxima = batch[self.maxima].data
        cell_indicator = batch[self.cell_indicator].data
        parent_vectors = batch[self.parent_vectors].data

        cells = []
        for index in np.argwhere(maxima*cell_indicator > self.score_threshold):
            index = gp.Coordinate(index)
            logger.debug("Getting parent vector at index %s" % str(index))

            score = cell_indicator[index]
            if self.edge_length == 1:
                parent_vector = tuple(
                    float(x) for x in parent_vectors[(Ellipsis,) + index])
            else:
                parent_vector = WriteCells.get_avg_pv(
                        parent_vectors, index, self.edge_length)
            position = roi.get_begin() + voxel_size*index
            if self.volume_shape is not None and \
               np.any(np.greater_equal(
                   position,
                   gp.Coordinate(self.volume_shape) * voxel_size)):
                continue

            cell_id = int(math.cantor_number(
                roi.get_begin()/voxel_size + index))

            cells.append({
                'id': cell_id,
                'score': float(score),
                't': position[0],
                'z': position[1],
                'y': position[2],
                'x': position[3],
                'parent_vector': parent_vector
            })

            logger.debug(
                "ID=%d, score=%f, parent_vector=%s" % (
                    cell_id, score, parent_vector))

        if len(cells) > 0:
            try:
                self.cells.insert_many(cells)
            except BulkWriteError as bwe:
                logger.error(bwe.details)
                raise


    def get_avg_pv(parent_vectors, index, edge_length):
        ''' Computes the average parent vector offset from the parent vectors
        in a cube centered at index. Accounts for the fact that each parent
        vector is a relative offset from its source location, not from index.

        Args:

            parent_vectors (``np.array``):

                A numpy array of parent vectors with dimensions
                (channels, time, z, y, x).

            index (``gp.Coordinate``):

                A 4D coordiante (t, z, y, x) indicating the target
                location to get the average parent vector for.

            edge_length (``int``):

                Length of each side of the cube within which the
                parent vectors are averaged.

        '''
        radius = (edge_length - 1) // 2
        logger.debug("Getting average parent vectors with radius"
                     " %d around index %s" % (radius, str(index)))
        offsets = []
        pv_shape = parent_vectors.shape
        # channels, t, z, y, x
        assert(len(pv_shape) == 5)
        pv_max_z = pv_shape[2]
        pv_max_y = pv_shape[3]
        pv_max_x = pv_shape[4]
        logger.debug("Type of index[1]: %s   index[1] %s"
                     % (str(type(index[1])), str(index[1])))
        for z in range(max(0, index[1] - radius),
                       min(index[1] + radius + 1, pv_max_z)):
            for y in range(max(0, index[2] - radius),
                           min(index[2] + radius + 1, pv_max_y)):
                for x in range(max(0, index[3] - radius),
                               min(index[3] + radius + 1, pv_max_x)):
                    c = gp.Coordinate((z, y, x))
                    c_with_time = gp.Coordinate((index[0], z, y, x))
                    relative_pos = c - index[1:]
                    offset_relative_to_c = parent_vectors[
                            (Ellipsis,) + c_with_time]
                    offsets.append(offset_relative_to_c + relative_pos)
        logger.debug("Offsets to average: %s" + str(offsets))
        parent_vector = tuple(float(sum(col) / len(col))
                              for col in zip(*offsets))
        return parent_vector
