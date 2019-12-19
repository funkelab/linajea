import gunpowder as gp
import numpy as np


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
