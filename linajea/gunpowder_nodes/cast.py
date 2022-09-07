"""Provides a gunpowder node to cast array from one type to another
"""
import copy

import gunpowder as gp
import numpy as np


class Cast(gp.BatchFilter):
    """Gunpowder node to cast array to dtype

    Attributes
    ----------
    array: gp.ArrayKey
        array to cast to new type
    dtype: np.dtype
        new dtype of array
    """
    def __init__(
            self,
            array,
            dtype=np.float32):

        self.array = array
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()
        array_spec = copy.deepcopy(self.spec[self.array])
        array_spec.dtype = self.dtype
        self.updates(self.array, array_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array]
        deps[self.array].dtype = None
        return deps

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.spec.dtype = self.dtype
        array.data = array.data.astype(self.dtype)
