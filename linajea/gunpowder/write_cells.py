from funlib import math
import gunpowder as gp
import numpy as np
import pymongo
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
            edge_length=1):
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

    def process(self, batch, request):

        if self.client is None:
            self.client = pymongo.MongoClient(host=self.db_host)
            self.db = self.client[self.db_name]
            create_indices = 'nodes' not in self.db.list_collection_names()
            self.cells = self.db['nodes']
            if create_indices:
                self.cells.create_index(
                    [
                        (l, pymongo.ASCENDING)
                        for l in ['t', 'z', 'y', 'x']
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
            radius = (self.edge_length - 1) / 2
            score = cell_indicator[index]
            if radius == 0:
                parent_vector = parent_vectors[(Ellipsis,) + index]
            else:
                parent_vector = self.get_avg_pv(parent_vectors, index, radius)
            logger.info("Parent vector: %s" % str(parent_vector))
            position = roi.get_begin() + voxel_size*index

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
            self.cells.insert_many(cells)

    def get_avg_pv(parent_vectors, index, radius):

        offsets = []
        for z in range(index[1] - radius, index[1] + radius + 1):
            for y in range(index[2] - radius, index[2] + radius + 1):
                for x in range(index[3] - radius, index[3] + radius + 1):
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
