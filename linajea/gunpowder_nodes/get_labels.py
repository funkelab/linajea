"""Provides a gunpowder node to set labels
"""
import numpy as np

import gunpowder as gp


class GetLabels(gp.BatchFilter):
    """Gunpowder node to extract label from attribute set on raw array

    Used in combination with gp.SpecifiedLocation, expects
    specified_location_extra_data attribute on raw gp.Array to be set

    Attributes
    ----------
    raw: gp.ArrayKey
        Input array containing raw data, needs to have
        specified_location_extra_data attribute set
    labels: gp.ArrayKey
        Output array
    """
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
