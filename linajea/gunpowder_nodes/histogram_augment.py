"""Provides a histogram augment gunpowder node
"""
import logging

import numpy as np

from gunpowder.batch_request import BatchRequest
from gunpowder.nodes.batch_filter import BatchFilter


logger = logging.getLogger(__name__)


class HistogramAugment(BatchFilter):
    '''Randomly transform the values of an intensity array by
    changing its histogram.
    Example: in/decrease peaks while keeping lower values mostly fixed

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        range_low (``float``):
        range_high (``float``):

            The min and max of the uniformly randomly drawn factor
            for the histogram augmentation. Intensities are
            changed as::

                shift = (1.0 - data) / (1 - 0)
                data = data * factor + (data * shift) * (1.0 - factor)

        z_section_wise (``bool``):

            Perform the augmentation z-section wise. Requires 3D arrays and
            assumes that z is the first dimension.
    '''

    def __init__(self, array, range_low, range_high, z_section_wise=False):
        self.array = array
        self.range_low = range_low
        self.range_high = range_high
        self.z_section_wise = z_section_wise

    def setup(self):
        logger.info("setting up histogram augment")
        assert self.range_low >= 0.0 and self.range_low <= 1.0 and \
            self.range_high >= 0.0 and self.range_high <= 1.0 and \
            self.range_low <= self.range_high, \
            "invalid range for histogram augment"
        self.updates(self.array, self.spec[self.array])

        self.enable_autoskip()

    def prepare(self, request):
        # TODO: move all randomness into the prepare method
        # TODO: write a test for this node
        np.random.seed(request.random_seed)
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        assert not self.z_section_wise or raw.spec.roi.dims() == 3, \
            "If you specify 'z_section_wise', I expect 3D data."
        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, \
            ("Intensity augmentation requires float types for the raw array "
             "(not " + str(raw.data.dtype) + "). Consider using Normalize "
             "before.")

        if self.z_section_wise:
            shape = raw.spec.roi/self.spec[self.array].voxel_size
            z_shape = shape.get_shape()[0]
            for z in range(z_shape):
                raw.data[z] = self.__augment(
                        raw.data[z],
                        np.random.uniform(self.range_low, self.range_high))
        else:
            raw.data = self.__augment(
                raw.data,
                np.random.uniform(self.range_low, self.range_high))

    def __augment(self, a, factor):
        shift = (1.0 - a) / (1 - 0)
        a = a * factor + (a * shift) * (1.0 - factor)
        return a
