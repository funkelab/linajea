cimport cython
import logging
import numpy as np

logger = logging.getLogger(__name__)

def target_counts(parent_vectors, voxel_size, mask=None):
    '''Given an array of parent vectors, counts for each voxel how many parent
    vectors point to it. Parent vectors are relative, i.e., an entry
    ``parent_vectors[:,z,y,x] == [0, 0, 0]`` points to ``[z,y,x]``,
    ``[1, 0, 0]`` points to ``[z+1,y,x]``.

    Args:

        parent_vectors (``ndarray``):

            Array of shape ``(3, d, h, w)`` containing parent vectors.

        voxel_size (``tuple`` of ``int``):

            The size of a voxel in world units.

        mask (``ndarray``, optional):

            A mask to count only offset vectors for voxels with value 1 in the
            mask.

    Returns:

        An ``ndarray`` of shape (d, h, w), containing the number of offset
        vectors pointing to the respective location.
    '''

    # make sure the inputs are all 3D
    assert len(parent_vectors.shape) == 4, (
        "parent_vectors needs to be a 4d array")
    assert parent_vectors.shape[0] == 3, (
        "parent vectors should be 3D vectors")
    assert len(voxel_size) == 3, "voxel size is not 3D"

    logger.debug(
        "counting targets in parent vectors of shape %s with voxel size %s",
        parent_vectors.shape, voxel_size)

    # (3, d, h, w)
    offset_vectors = np.array(parent_vectors)

    # convert offsets from world units to voxels
    offset_vectors[0] /= voxel_size[0]
    offset_vectors[1] /= voxel_size[1]
    offset_vectors[2] /= voxel_size[2]

    # discretize offset vectors
    offset_vectors = np.round(offset_vectors).astype(np.int32)

    return target_counts_from_offsets(offset_vectors, mask)

@cython.boundscheck(False)
@cython.wraparound(False)
def target_counts_from_offsets(offsets, mask=None):
    '''Given a volume of offsets, counts for each voxel how many offsets point
    to it. Offsets are relative, i.e., an entry ``offsets[:,z,y,x] == [0, 0,
    0]`` points to ``[z,y,x]``, ``[1, 0, 0]`` points to ``[z+1,y,x]``.

    Args:

        offsets (``ndarray``):

            Array of shape ``(3, d, h, w)`` containing offset vectors.

        mask (``ndarray``, optional):

            A mask to count only offset vectors for voxels with value 1 in the
            mask.

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
    cdef int [:,:,:] targets_mask = mask

    # count number of indices
    for z in range(d):
        for y in range(h):
            for x in range(w):
                if mask is None or mask[z, y, x] == 1:
                    tz = min(max(0, z + targets[0, z, y, x]), d - 1)
                    ty = min(max(0, y + targets[1, z, y, x]), h - 1)
                    tx = min(max(0, x + targets[2, z, y, x]), w - 1)
                    counts[tz, ty, tx] += 1

    return counts_np
