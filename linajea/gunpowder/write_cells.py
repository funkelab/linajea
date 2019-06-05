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
            db_name):

        self.maxima = maxima
        self.cell_indicator = cell_indicator
        self.parent_vectors = parent_vectors
        self.score_threshold = score_threshold
        self.db_host = db_host
        self.db_name = db_name
        self.client = None

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

            score = cell_indicator[index]
            parent_vector = parent_vectors[(Ellipsis,) + index]
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
                'parent_vector': tuple(float(x) for x in parent_vector)
            })

            logger.debug(
                "ID=%d, score=%f, parent_vector=%s" % (
                    cell_id, score, parent_vector))

        if len(cells) > 0:
            self.cells.insert_many(cells)
