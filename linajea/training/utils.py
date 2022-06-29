import copy
import glob
import math
import re

import numpy as np

import gunpowder as gp


def get_latest_checkpoint(basename):

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    checkpoints = glob.glob(basename + '_checkpoint_*')
    checkpoints.sort(key=natural_keys)

    if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
        iteration = int(checkpoint.split('_')[-1].split('.')[0])
        return checkpoint, iteration

    return None, 0


class Cast(gp.BatchFilter):

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


def crop(x, shape):
    '''Center-crop x to match spatial dimensions given by shape.'''

    dims = len(x.size()) - len(shape)
    x_target_size = x.size()[:dims] + shape

    offset = tuple(
        (a - b)//2
        for a, b in zip(x.size(), x_target_size))

    slices = tuple(
        slice(o, o + s)
        for o, s in zip(offset, x_target_size))

    # print(x.size(), shape, slices)
    return x[slices]


def crop_to_factor(x, factor, kernel_sizes):
    '''Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.

    The crop could be done after the convolutions, but it is more efficient
    to do that before (feature maps will be smaller).
    '''

    shape = x.size()
    dims = len(x.size()) - 2
    spatial_shape = shape[-dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = tuple(
        sum(ks[d] - 1 for ks in kernel_sizes)
        for d in range(dims)
    )

    # we need (spatial_shape - convolution_crop) to be a multiple of
    # factor, i.e.:
    #
    # (s - c) = n*k
    #
    # we want to find the largest n for which s' = n*k + c <= s
    #
    # n = floor((s - c)/k)
    #
    # this gives us the target shape s'
    #
    # s' = n*k + c

    ns = (
        int(math.floor(float(s - c)/f))
        for s, c, f in zip(spatial_shape, convolution_crop, factor)
    )
    target_spatial_shape = tuple(
        n*f + c
        for n, c, f in zip(ns, convolution_crop, factor)
    )

    if target_spatial_shape != spatial_shape:

        assert all((
            (t > c) for t, c in zip(
                target_spatial_shape,
                convolution_crop))
                   ), \
                   "Feature map with shape %s is too small to ensure " \
                   "translation equivariance with factor %s and following " \
                   "convolutions %s" % (
                       shape,
                       factor,
                       kernel_sizes)

        return crop(x, target_spatial_shape)

    return x
