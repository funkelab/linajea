"""Provides gunpowder nodes for data clipping and normalization
"""
import gunpowder as gp
import numpy as np


class Clip(gp.BatchFilter):
    """Gunpowder node to clip data in array to range

    Attributes
    ----------
    array: gp.ArrayKey
        data to clip
    mn: float
        lower bound
    mx: float
        upper bound
    """
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


class NormalizeAroundZero(gp.Normalize):
    """Gunpowder node to normalize data in array with mean and std

    Attributes
    ----------
    array: gp.ArrayKey
        data to normalize
    mapped_to_zero: float
        pixels with this value be mapped to 0 (typically mean)
    diff_mapped_to_one: float
        pixels with an absolute difference of this value to the
        mapped_to_zero value will be mapped to +-1 (typically std)
    dtype: np.dtype
        cast output array to this type
    """
    def __init__(
            self,
            array,
            mapped_to_zero,
            diff_mapped_to_one,
            dtype=np.float32):

        self.array = array
        self.mapped_to_zero = mapped_to_zero
        self.diff_mapped_to_one = diff_mapped_to_one
        self.dtype = dtype

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.spec.dtype = self.dtype
        array.data = array.data.astype(self.dtype)
        array.data = (array.data - self.mapped_to_zero) / self.diff_mapped_to_one
        array.data = array.data.astype(self.dtype)


class NormalizeLowerUpper(NormalizeAroundZero):
    """Gunpowder node to normalize data in array from range to [0, 1]

    Attributes
    ----------
    array: gp.Array
        data to normalize
    lower: float
        lower bound, will be mapped to 0
    upper: float
        upper bound, will be mapped to 1
    dtype: np.dtype
        cast output array to this type
    """
    def __init__(
            self,
            array,
            lower,
            upper,
            interpolatable=True,
            dtype=np.float32):

        super(NormalizeLowerUpper, self).__init__(array, lower, upper-lower)
