"""Utility functions for training process
"""
import glob
import logging
import math
import re

import numpy as np

import gunpowder as gp

from linajea.gunpowder_nodes import (Clip,
                                     NormalizeAroundZero,
                                     NormalizeLowerUpper)

logger = logging.getLogger(__name__)


def get_latest_checkpoint(basename):
    """Looks for the checkpoint with the highest step count

    Checks for files name basename + '_checkpoint_*'
    The suffix should be the iteration count
    Selects the one with the highest one and returns the path to it
    and the step count

    Args
    ----
    basename: str
        Path to and prefix of model checkpoints

    Returns
    -------
    2-tuple: str, int
        Path to and iteration of latest checkpoint

    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    checkpoints = glob.glob(basename + '_checkpoint_*')
    checkpoints.sort(key=natural_keys)

    if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
        iteration = int(checkpoint.split('_')[-1].split('.')[0])
        return checkpoint, iteration

    return None, 0


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


def normalize(pipeline, normalization, raw, data_config=None):
    """Add data normalization node to pipeline

    Args
    ----
    pipeline: gp.BatchFilter
        Gunpowder node/pipeline, typically a source node containing data
        that should be normalized
    normalization: NormalizeConfig
        Normalization configuration object used to determine which type of
        normalization should be performed
    raw: gp.ArrayKey
        Key identifying which array to normalize
    data_config: dict of str: int, optional
        Object containing statistics about data set that can be used
        to normalize data

    Returns
    -------
    gp.BatchFilter
        Pipeline extended by a normalization node

    Notes
    -----
    Which normalization method should be used?
    None/default:
        [0,1] based on data type
    minmax:
        normalize such that lower bound is at 0 and upper bound at 1
        clipping is less strict, some data might be outside of range
    percminmax:
        use precomputed percentile values for minmax normalization;
        precomputed values are stored in data_config file that has to
        be supplied; set perc_min/max to tag to be used
    mean/median
        normalize such that mean/median is at 0 and 1 std/mad is at -+1
        set perc_min/max tags for clipping beforehand
    """
    if normalization is None or \
       normalization.type == 'default':
        logger.info("default normalization")
        pipeline = pipeline + \
            gp.Normalize(raw,
                         factor=1.0/np.iinfo(data_config['stats']['dtype']).max
                         if data_config is not None else None)
    elif normalization.type == 'minmax':
        mn = normalization.norm_bounds[0]
        mx = normalization.norm_bounds[1]
        logger.info("minmax normalization %s %s", mn, mx)
        pipeline = pipeline + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeLowerUpper(raw, lower=mn, upper=mx, interpolatable=False)
    elif normalization.type == 'percminmax':
        mn = data_config['stats'][normalization.perc_min]
        mx = data_config['stats'][normalization.perc_max]
        logger.info("perc minmax normalization %s %s", mn, mx)
        pipeline = pipeline + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeLowerUpper(raw, lower=mn, upper=mx)
    elif normalization.type == 'mean':
        mean = data_config['stats']['mean']
        std = data_config['stats']['std']
        mn = data_config['stats'][normalization.perc_min]
        mx = data_config['stats'][normalization.perc_max]
        logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
        pipeline = pipeline + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeAroundZero(raw, mapped_to_zero=mean,
                                diff_mapped_to_one=std)
    elif normalization.type == 'median':
        median = data_config['stats']['median']
        mad = data_config['stats']['mad']
        mn = data_config['stats'][normalization.perc_min]
        mx = data_config['stats'][normalization.perc_max]
        logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
        pipeline = pipeline + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeAroundZero(raw, mapped_to_zero=median,
                                diff_mapped_to_one=mad)
    else:
        raise RuntimeError("invalid normalization method %s",
                           normalization.type)
    return pipeline
