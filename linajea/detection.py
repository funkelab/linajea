from .nms import find_maxima, sphere
from .target_counts import target_counts
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
from skimage.measure import block_reduce
import logging
import math
import numpy as np
import peach
import time

logger = logging.getLogger(__name__)

class CellDetectionParameters(object):

    def __init__(
            self,
            nms_radius,
            sigma=None,
            downsample=None,
            min_score_threshold=0):

        self.nms_radius = nms_radius
        self.sigma = sigma
        self.downsample = downsample
        self.min_score_threshold = 0

class EdgeDetectionParameters(object):
    '''
    Args:

        move_threshold (``float``):

            By how much cells are allowed to move spatially between frames.

        pool_radius (``tuple`` of ``int``):

            The 3D radius of a cell to consider to pool parent vectors from. The
            parent vectors within radius will be used to compute the target
            counts between two cells, i.e, how many parent vectors from one cell
            point to the center of the other cell.

        sigma (``float``):

            By how much to smooth the target counts. This is equivalent to
            saying that each parent vector produces a Gaussian in the target
            frame. The score of an edge from u to v is the sum of all the
            Gaussians at the center of u.
    '''

    def __init__(self, move_threshold, pool_radius, sigma):

        self.move_threshold = move_threshold
        self.pool_radius = pool_radius
        self.sigma = sigma

def find_cells(
        target_counts,
        parameters):

    if parameters.downsample:

        downsample = tuple(parameters.downsample)

        print("Downsampling target_counts...")
        start = time.time()
        downsampled = block_reduce(target_counts.data, (1,) + downsample, np.sum)
        voxel_size = target_counts.voxel_size*peach.Coordinate((1,) + downsample)
        target_counts = peach.Array(
            downsampled,
            peach.Roi(
                target_counts.roi.get_begin(),
                voxel_size*downsampled.shape),
            voxel_size)
        print("%.3fs"%(time.time()-start))
        print("new voxel size of target_counts: %s"%(target_counts.voxel_size,))

    centers, labels, target_counts_smoothed = find_maxima(
        target_counts,
        (0.1,) + parameters.nms_radius, # 0.1 == no NMS over t
        (0,) + parameters.sigma,
        parameters.min_score_threshold)

    return centers, labels, target_counts_smoothed

def find_edges(
        parent_vectors,
        cells,
        parameters):
    '''Find and score edges between cells.

    Args:

        parent_vectors (``peach.Array``):

            An array of predicted parent vectors.

        cells (``dict``):

            Dict from ``id: center`` of each cell.

        parameters (`class:EdgeDetectionParameters`):

            Parameters of the edge detection, see there for details.
    '''

    assert parent_vectors.roi.dims() == 4, (
        "Expect 4D input.")

    t_begin = parent_vectors.roi.get_begin()[0]
    t_end = parent_vectors.roi.get_end()[0]

    # sort cells by frame
    cells_by_t = {
        t: [
            label
            for label, center in cells.items()
            if center[0] == t
        ]
        for t in range(t_begin, t_end)
    }

    logger.debug("Got cells in frames %s", cells_by_t.keys())

    voxel_size_3d = parent_vectors.voxel_size[1:]

    # create a 3D mask centered at (0, 0, 0) to pool parent vectors from
    radius_vx = peach.Coordinate(
        int(math.ceil(r/v))
        for r, v in zip(parameters.pool_radius, voxel_size_3d))
    shape = radius_vx*2 + (1, 1, 1)
    mask_roi = peach.Roi(
        (0, 0, 0),
        shape*voxel_size_3d)
    mask_roi -= mask_roi.get_center()
    mask = peach.Array(
        sphere(radius_vx).astype(np.int32),
        mask_roi,
        voxel_size_3d)

    edges = []
    for t in range(t_begin, t_end - 1):

        pre = t
        nex = t + 1

        # prepare KD tree for fast partner lookup
        nex_ids = np.array(cells_by_t[nex])
        kd_data = [ cells[cell_id][1:] for cell_id in nex_ids ]

        if len(nex_ids) == 0:
            continue

        nex_kd_tree = KDTree(kd_data)

        logger.debug(
            "Finding edges between cells in frames %d and %d (%d and %d cells)",
            pre, nex, len(cells_by_t[pre]), len(cells_by_t[nex]))

        for pre_cell in cells_by_t[pre]:

            print(pre_cell)

            nex_neighbor_indices = nex_kd_tree.query_ball_point(
                cells[pre_cell][1:],
                parameters.move_threshold)
            nex_neighbors = nex_ids[nex_neighbor_indices]

            for nex_cell in nex_neighbors:

                pre_center = np.array(cells[pre_cell][1:])
                nex_center = np.array(cells[nex_cell][1:])

                moved = (pre_center - nex_center)
                distance = np.linalg.norm(moved)

                # Get score from nex to pre (backwards in time).
                #
                # Cut out parent vectors around mask_roi centered at nex_center.
                # We set the fill_value to 1000 to make sure that out-of-bounds
                # voxels don't contribute to the target counts (they will point
                # very far away).

                # get the mask ROI around the next cell in 3D
                nex_roi_3d = mask_roi + peach.Coordinate(nex_center)
                # add the time dimension
                nex_roi_4d = peach.Roi(
                    (nex,) + nex_roi_3d.get_begin(),
                    (1,) + nex_roi_3d.get_shape())
                nex_parent_vectors = parent_vectors.fill(
                    nex_roi_4d,
                    fill_value=1000)

                # get smoothed target counts at pre
                assert nex_parent_vectors.shape[1] == 1
                counts = target_counts(
                    nex_parent_vectors.data[:,0,:].astype(np.int32),
                    mask.data)
                assert len(counts.shape) == 3
                counts = gaussian_filter(
                    counts,
                    parameters.sigma,
                    mode='constant')
                counts = peach.Array(
                    counts,
                    nex_roi_3d,
                    voxel_size_3d)
                score = counts[counts.roi.get_center()]

                edges.append({
                    'source': int(nex_cell),
                    'target': int(pre_cell),
                    'score': float(score),
                    'distance': float(distance)
                })

    return edges
