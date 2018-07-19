cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def target_counts(offsets):
    '''Given a volume of offsets, counts for each voxel how many offsets point
    to it. Offsets are relative, i.e., an entry ``offsets[:,z,y,x] == [0, 0,
    0]`` points to ``[z,y,x]``, ``[1, 0, 0]`` points to ``[z+1,y,x]``.

    Args:

        offsets (``ndarray``):

            Array of shape ``(3, d, h, w)`` containing offset vectors.

    Returns:

        An ndarray of shape (d, h, w), containing the number of offset vectors
        pointing to the respective location.
    '''

    cdef Py_ssize_t d, h, w, z, y, x, tz, ty, tx

    d = offsets.shape[1]
    h = offsets.shape[2]
    w = offsets.shape[3]

    # numpy array and C memory view on counts:
    counts_np = np.zeros((d, h, w), dtype=np.intc)
    cdef int [:,:,:] counts = counts_np

    cdef int [:,:,:,:] targets = offsets

    # count number of indices
    for z in range(d):
        for y in range(h):
            for x in range(w):
                tz = min(max(0, z + targets[0, z, y, x]), d - 1)
                ty = min(max(0, y + targets[1, z, y, x]), h - 1)
                tx = min(max(0, x + targets[2, z, y, x]), w - 1)
                counts[tz, ty, tx] += 1

    return counts_np
