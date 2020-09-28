import gunpowder as gp
import numpy as np


class Clip(gp.BatchFilter):

    def __init__(self, array, mn=None, mx=None):

        self.array = array
        self.mn = mn
        self.mx = mx

    def process(self, batch, request):

        if self.mn is None and self.mx is None:
            return

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]


        if self.mn is None:
            self.mn = np.min(array.data)
        if self.mx is None:
            self.mx = np.max(array.data)

        array.data = np.clip(array.data, self.mn, self.mx)


class NormalizeMinMax(gp.BatchFilter):

    def __init__(
            self,
            array,
            mn,
            mx,
            dtype=np.float32,
            clip=False):

        self.array = array
        self.mn = mn
        self.mx = mx
        self.dtype = dtype
        self.clip = clip

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.data = array.data.astype(self.dtype)
        if self.clip:
            array.data = np.clip(array.data, self.mn, self.mx)
        array.data = (array.data - self.mn) / (self.mx - self.mn)
        array.data = array.data.astype(self.dtype)

class NormalizeMeanStd(gp.BatchFilter):

    def __init__(
            self,
            array,
            mean,
            std,
            dtype=np.float32):

        self.array = array
        self.mean = mean
        self.std = std
        self.dtype = dtype

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.data = array.data.astype(self.dtype)
        array.data = (array.data - self.mean) / self.std
        array.data = array.data.astype(self.dtype)

class NormalizeMedianMad(gp.BatchFilter):

    def __init__(
            self,
            array,
            median,
            mad,
            dtype=np.float32):

        self.array = array
        self.median = median
        self.mad = mad
        self.dtype = dtype

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.data = array.data.astype(self.dtype)
        array.data = (array.data - self.median) / self.mad
        array.data = array.data.astype(self.dtype)
