from .nms import find_maxima, sphere
from skimage.measure import block_reduce
import logging
import math
import numpy as np
import time

logger = logging.getLogger(__name__)

def find_cells(
        target_counts,
        voxel_size,
        radius,
        sigma=None,
        min_score_threshold=0,
        downsample=None):

    if downsample:
        downsample = tuple(downsample)
        print("Downsampling target_counts...")
        start = time.time()
        # TODO: sum would make more sense! test it
        target_counts = block_reduce(target_counts, (1,) + downsample, np.max)
        voxel_size = tuple(r*d for r, d in zip(voxel_size, (1,) + downsample))
        print("%.3fs"%(time.time()-start))
        print("new voxel size of target_counts: %s"%(voxel_size,))

    centers, labels, target_counts = find_maxima(
        target_counts,
        voxel_size,
        (0.1,) + radius, # 0.1 == no NMS over t
        (0,) + sigma,
        min_score_threshold)

    return centers, labels, target_counts, voxel_size

def find_edges(
        parent_vectors,
        voxel_size,
        cell_centers,
        move_threshold,
        radius,
        sigma):

    cells_by_t = {
        t: [
            label
            for label, d in cell_centers.items()
            if d['center'][0] == t
        ]
        for t in range(parent_vectors.shape[0])
    }

    radius_vx = tuple(
        int(math.ceil(r*v))
        for r, v in zip(radius, voxel_size))
    mask = sphere(radius_vx)

    logger.debug("Got cells in frames %s", cells_by_t.keys())

    edges = []
    logger.debug("Shape of parent_vectors: %s", parent_vectors.shape)
    for t in range(0, parent_vectors.shape[1] - 1):

        pre = t
        nex = t + 1

        logger.debug(
            "Finding edges between cells in frames %d and %d",
            pre, nex)

        for pre_cell in cells_by_t[pre]:
            for nex_cell in cells_by_t[nex]:

                pre_center = np.array(cell_centers[pre_cell]['center'])
                nex_center = np.array(cell_centers[pre_cell]['center'])

                moved = (pre_center - nex_center)*voxel_size
                distance = np.linalg.norm(moved)

                logger.debug(
                    "Considering edge between %s at %s and %s at %s "
                    "with distance %f",
                    pre_cell, pre_center, nex_cell, nex_center, distance)

                if distance > move_threshold:
                    logger.debug("distance exceeds threshold")
                    continue

                # get score from nex to pre (backwards in time)

                # cut out parent vectors around nex
                nex_center = tuple(int(c) for c in nex_center)
                slices = tuple(
                    slice(c - r, c + r)
                    for c, r in zip(nex_center, radius_vx)
                )
                nex_parent_vectors = parent_vectors[(slice(t + 1),) + slices]

                assert nex_parent_vectors.shape == mask.shape

                # get smoothed target counts at pre
                counts = target_counts(nex_parent_vectors, mask)
                counts = gaussian_filter(counts, sigma, mode='constant')
                score = counts[radius_vx[0], radius_vx[1], radius_vx[2]]

                edges.append({
                    'from': nex_cell,
                    'to': pre_cell,
                    'score': score
                })

    return edges
