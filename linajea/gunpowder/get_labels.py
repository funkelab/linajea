import numpy as np

import gunpowder as gp


class GetLabels(gp.BatchFilter):

    def __init__(self, raw, labels):

        self.raw = raw
        self.labels = labels

    def setup(self):

        self.provides(self.labels, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        pass

    def process(self, batch, request):
        batch.arrays[self.labels] = \
            gp.Array(np.asarray(batch.arrays[self.raw].attrs[
                'specified_location_extra_data'],
                                dtype=request[self.labels].dtype),
                     spec=request[self.labels])
