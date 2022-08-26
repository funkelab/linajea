"""Provides a gunpowder node to write node indicators and movement vectors to
database
"""
import logging
import pymongo
from pymongo.errors import BulkWriteError

import numpy as np

from funlib import math
import gunpowder as gp

logger = logging.getLogger(__name__)


class WriteCells(gp.BatchFilter):
    """Gunpowder node to write tracking prediction data to database

    Attributes
    ----------
    maxima: gp.ArrayKey
        binary array containing extracted maxima
    cell_indicator: gp.ArrayKey
        array containing cell indicator prediction
    movement_vectors: gp.ArrayKey
        array containing movement vector prediction
    score_threshold: float
        ignore maxima with a cell indicator score lower than this
    db_host: str
        mongodb host
    db_name: str
        write to this database
    edge_length: int
        if > 1, edge length indicates the length of the edge of a cube
        from which movement vectors will be read. The cube will be
        centered around the maxima, and predictions within the cube
        of voxels will be averaged to get the movement vector to store
        in the db
    mask: np.ndarray
        If not None, use as mask and ignore all predictions outside of
        mask
    z_range: 2-tuple
        If not None, ignore all predictions ouside of the given
        z/depth range
    volume_shape: gp.Coordinate or list of int
        If not None, should be set to shape of volume (in voxels);
        ignore all predictions outside (might occur if using daisy with
        overhang)
    """
    def __init__(
            self,
            maxima,
            cell_indicator,
            movement_vectors,
            score_threshold,
            db_host,
            db_name,
            edge_length=1,
            mask=None,
            z_range=None,
            volume_shape=None):

        self.maxima = maxima
        self.cell_indicator = cell_indicator
        self.movement_vectors = movement_vectors
        self.score_threshold = score_threshold
        self.db_host = db_host
        self.db_name = db_name
        self.client = None
        assert edge_length % 2 == 1, "Edge length should be odd"
        self.edge_length = edge_length
        self.mask = mask
        self.z_range = z_range
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
        movement_vectors = batch[self.movement_vectors].data

        cells = []
        for index in np.argwhere(maxima*cell_indicator > self.score_threshold):
            index = gp.Coordinate(index)
            logger.debug("Getting movement vector at index %s" % str(index))

            score = cell_indicator[index]
            if self.edge_length == 1:
                movement_vector = tuple(
                    float(x) for x in movement_vectors[(Ellipsis,) + index])
            else:
                movement_vector = WriteCells.get_avg_mv(
                        movement_vectors, index, self.edge_length)
            position = roi.get_begin() + voxel_size*index
            if self.volume_shape is not None and \
               np.any(np.greater_equal(
                   position,
                   gp.Coordinate(self.volume_shape) * voxel_size)):
                continue

            if self.mask is not None:
                tmp_pos = position // voxel_size
                if self.mask[tmp_pos[-self.mask.ndim:]] == 0:
                    logger.debug("skipping cell mask {}".format(tmp_pos))
                    continue
            if self.z_range is not None:
                tmp_pos = position // voxel_size
                if tmp_pos[1] < self.z_range[0] or \
                   tmp_pos[1] > self.z_range[1]:
                    logger.debug("skipping cell zrange {}".format(tmp_pos))
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
                'movement_vector': movement_vector
            })

            logger.debug(
                "ID=%d, score=%f, movement_vector=%s" % (
                    cell_id, score, movement_vector))

        if len(cells) > 0:
            try:
                self.cells.insert_many(cells)
            except BulkWriteError as bwe:
                logger.error(bwe.details)
                raise

    def get_avg_mv(movement_vectors, index, edge_length):
        ''' Computes the average movement vector offset from the movement vectors
        in a cube centered at index. Accounts for the fact that each movement
        vector is a relative offset from its source location, not from index.

        Args:

            movement_vectors (``np.array``):

                A numpy array of movement vectors with dimensions
                (channels, time, z, y, x).

            index (``gp.Coordinate``):

                A 4D coordiante (t, z, y, x) indicating the target
                location to get the average movement vector for.

            edge_length (``int``):

                Length of each side of the cube within which the
                movement vectors are averaged.

        '''
        radius = (edge_length - 1) // 2
        logger.debug("Getting average movement vectors with radius"
                     " %d around index %s" % (radius, str(index)))
        offsets = []
        mv_shape = movement_vectors.shape
        # channels, t, z, y, x
        assert(len(mv_shape) == 5)
        mv_max_z = mv_shape[2]
        mv_max_y = mv_shape[3]
        mv_max_x = mv_shape[4]
        logger.debug("Type of index[1]: %s   index[1] %s"
                     % (str(type(index[1])), str(index[1])))
        for z in range(max(0, index[1] - radius),
                       min(index[1] + radius + 1, mv_max_z)):
            for y in range(max(0, index[2] - radius),
                           min(index[2] + radius + 1, mv_max_y)):
                for x in range(max(0, index[3] - radius),
                               min(index[3] + radius + 1, mv_max_x)):
                    c = gp.Coordinate((z, y, x))
                    c_with_time = gp.Coordinate((index[0], z, y, x))
                    relative_pos = c - gp.Coordinate(index[1:])
                    offset_relative_to_c = movement_vectors[
                            (Ellipsis,) + c_with_time]
                    offsets.append(offset_relative_to_c + relative_pos)
        logger.debug("Offsets to average: %s" + str(offsets))
        movement_vector = tuple(float(sum(col) / len(col))
                                for col in zip(*offsets))
        return movement_vector
