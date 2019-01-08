import daisy
import logging
import numpy as np
import time
import operator
import functools

logger = logging.getLogger(__name__)

def downsample(parent_vectors, downsample_factors):
    """ Downsample a set of parent vectors by finding the coordinate-wise median
    Args:
        parent_vectors: A daisy array of parent vectors
        downsample_factors: A list of integers indicating the amount to downsample by
            in each of the space dimensions (e.g. [z, y, x])

    Returns: A daisy array where the voxel_size is multiplied by the downsample_factors
        and the data is pooled by finding the coordinate-wise median

    """
    assert len(downsample_factors) == 3
    assert len(parent_vectors.shape) == 5
    logger.info("Shape of parent vectors: {}".format(parent_vectors.shape))
    zyx_shape = parent_vectors.shape[2:]
    original_data = np.copy(parent_vectors.to_ndarray())
    channel_slice = slice(0, 3)
    time_slice = slice(0, parent_vectors.shape[1])
    num_slices = functools.reduce(operator.mul,
            daisy.Coordinate(parent_vectors.shape[2:]) // daisy.Coordinate(downsample_factors),
            1)
    count = 0
    st = time.time()
    logger.info("Starting pooling step")
    for z in range(0, zyx_shape[0], downsample_factors[0]):
        z_slice = slice(z, z + downsample_factors[0])
        for y in range(0, zyx_shape[1], downsample_factors[1]):
            y_slice = slice(y, y + downsample_factors[1])
            for x in range(0, zyx_shape[2], downsample_factors[2]):
                x_slice = slice(x, x + downsample_factors[2])
                s = (channel_slice, time_slice, z_slice, y_slice, x_slice)
               # logger.debug("Slice: {}".format(s))
                medians = np.median(original_data[s], axis=[2, 3, 4])
                original_data[s] = medians.reshape(parent_vectors.shape[0:2] + (1,1,1))
                count += 1
                if count % 10000 == 0:
                    logger.info("Finished {:.2f}% of downsample operations ({}/{}) in {} seconds"\
                            .format(count / float(num_slices), count, num_slices, time.time() - st))
                    
    downsampled_data= original_data[:,:,0:zyx_shape[0]:downsample_factors[0],
            0:zyx_shape[1]:downsample_factors[1],
            0:zyx_shape[2]:downsample_factors[2]]
    expected_shape = tuple(parent_vectors.shape[0:2]) +\
            tuple((p + d - 1) // d for p, d in zip(parent_vectors.shape[2:], downsample_factors))

    assert downsampled_data.shape == expected_shape, "Expected downsampled data to have shape {}, got {}"\
            .format(expected_shape, downsampled_data.shape)
    output_voxel_size = parent_vectors.voxel_size * daisy.Coordinate((1,) + tuple(downsample_factors))
    logger.info("Output voxel size: {}".format(output_voxel_size))
    output_roi = parent_vectors.roi.snap_to_grid(output_voxel_size)
    logger.debug("Output roi: {}".format(output_roi))
    downsampled_array = daisy.Array(downsampled_data,
                                    output_roi,
                                    output_voxel_size)
    
    return downsampled_array 

