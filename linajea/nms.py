from scipy.ndimage import measurements, label, maximum_filter
from scipy.ndimage.filters import gaussian_filter
import math
import numpy as np
import peach
import time

def find_maxima(
        array,
        radius,
        sigma=None,
        min_score_threshold=0):
    '''Find all points that are maximal within a sphere of ``radius`` and are
    strictly higher than min_score_threshold. Optionally smooth the prediction
    with sigma.

    Args:

        array (`class:peach.Array`):

            The array to find maxima in.

        radius (``tuple`` of ``int``):

            The radius (in world units) of a sphere to find maxima within.

        sigma (``tuple`` of ``float``, optional):

            By how much to smooth the values in ``array`` before finding maxima.

        min_score_threshold (``float``, optional):

            Only consider maxima with a value grater than this threshold.

    Returns:

        ``(centers, labels, array)``

        ``centers`` a dictionary from maxima IDs to a dictionary with 'center'
        and 'score', ``labels`` an array of the maximal points with their IDs,
        and ``array`` the (possibly smoothed) input array.
        '''

    values = array.data

    # smooth values
    if sigma is not None:
        print("Smoothing values...")
        sigma = tuple(
            float(s)/r
            for s, r in zip(sigma, array.voxel_size))
        print("voxel-sigma: %s"%(sigma,))
        start = time.time()
        values = gaussian_filter(
            values,
            sigma,
            mode='constant')
        print("%.3fs"%(time.time()-start))

    print("Finding maxima...")
    start = time.time()
    radius = tuple(
        int(math.ceil(float(ra)/re))
        for ra, re in zip(radius, array.voxel_size))
    print("voxel-radius: %s"%(radius,))
    max_filtered = maximum_filter(values, footprint=sphere(radius))

    maxima = max_filtered == values
    print("%.3fs"%(time.time()-start))

    print("Applying NMS...")
    start = time.time()
    values_filtered = np.zeros_like(values)
    values_filtered[maxima] = values[maxima]
    print("%.3fs"%(time.time()-start))

    print("Finding blobs...")
    start = time.time()
    blobs = values_filtered > min_score_threshold
    labels, num_blobs = label(blobs, output=np.uint64)
    print("%.3fs"%(time.time()-start))

    print("Found %d points after NMS"%num_blobs)

    print("Finding centers, sizes, and maximal values...")
    start = time.time()
    label_ids = np.arange(1, num_blobs + 1)
    centers = measurements.center_of_mass(blobs, labels, index=label_ids)
    maxima = measurements.maximum(values, labels, index=label_ids)
    print("%.3fs"%(time.time()-start))

    centers = {
        label: {
            'center': np.array([
                int(np.round(c*v + o))
                for c, v, o in zip(center, array.voxel_size, array.roi.get_offset())
            ]),
            'score': max_value
        }
        for label, center, max_value in zip(label_ids, centers, maxima)
    }

    labels = peach.Array(labels, array.roi, array.voxel_size)
    values = peach.Array(values, array.roi, array.voxel_size)

    return (centers, labels, values)

def sphere(radius):

    grid = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
    dist = sum([
        a.astype(np.float)**2/r**2
        for a, r in zip(grid, radius)
    ])
    return (dist <= 1)
