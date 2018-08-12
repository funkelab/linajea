from .nms import find_maxima, sphere
from .target_counts import target_counts
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
from skimage.measure import block_reduce
import logging
import math
import numpy as np
import daisy
import time

logger = logging.getLogger(__name__)

class CellDetectionParameters(object):
    '''
    Args:

        nms_radius (``tuple`` of ``int``):

            The 3D radius (in world units) to use for non-maxima suppression.

        sigma (``float``, optional):

            By how much to smooth the target counts (in world units) before
            searching for maxima.

        downsample (``tuple`` of ``int``, optional):

            By how much to downsample the target counts before searching for
            maxima.

        min_score_threshold (``float``, optional):

            Only consider maxima with a value grater than this threshold as
            cells.
    '''

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
        voxel_size = target_counts.voxel_size*daisy.Coordinate((1,) + downsample)
        target_counts = daisy.Array(
            downsampled,
            daisy.Roi(
                target_counts.roi.get_begin(),
                voxel_size*downsampled.shape),
            voxel_size)
        print("%.3fs"%(time.time()-start))
        print("new voxel size of target_counts: %s"%(target_counts.voxel_size,))

    # prepare output datastructures
    centers = {}
    labels = daisy.Array(
        np.zeros(target_counts.shape, dtype=np.uint64),
        target_counts.roi,
        target_counts.voxel_size)
    target_counts_smoothed = daisy.Array(
        np.zeros(target_counts.shape, dtype=np.float32),
        target_counts.roi,
        target_counts.voxel_size)

    # find maxima for each frame individually (we don't want maxima in one frame
    # shadowing maxima in another one)

    max_label = 0
    for t in range(
            target_counts.roi.get_begin()[0],
            target_counts.roi.get_end()[0]):

        frame_roi = target_counts.roi.intersect(
            daisy.Roi((t, None, None, None), (1, None, None, None)))

        frame_centers, frame_labels, frame_target_counts = find_maxima(
            target_counts[frame_roi],
            (0,) + parameters.nms_radius, # no NMS over t
            (0,) + parameters.sigma, # no smoothing over t
            parameters.min_score_threshold)

        # ensure unique IDs
        labels.data[labels.data!=0] += max_label
        frame_centers = {
            cell_id + max_label: value
            for cell_id, value in frame_centers.items()
        }
        max_label = np.max(labels.data)

        centers.update(frame_centers)
        labels[frame_roi] = frame_labels
        target_counts_smoothed[frame_roi] = frame_target_counts

    return centers, labels, target_counts_smoothed

def find_edges(
        parent_vectors,
        cells,
        parameters):
    '''Find and score edges between cells.

    Args:

        parent_vectors (``daisy.Array``):

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
            cell
            for cell in cells
            if cell['position'][0] == t
        ]
        for t in range(t_begin, t_end)
    }

    logger.debug("Got cells in frames %s", cells_by_t.keys())

    voxel_size_3d = parent_vectors.voxel_size[1:]

    # create a 3D mask centered at (0, 0, 0) to pool parent vectors from
    radius_vx = daisy.Coordinate(
        int(math.ceil(r/v))
        for r, v in zip(parameters.pool_radius, voxel_size_3d))
    shape = radius_vx*2 + (1, 1, 1)
    mask_roi = daisy.Roi(
        (0, 0, 0),
        shape*voxel_size_3d)
    mask_roi -= mask_roi.get_center()
    mask = daisy.Array(
        sphere(radius_vx).astype(np.int32),
        mask_roi,
        voxel_size_3d)

    edges = []
    for t in range(t_begin, t_end - 1):

        pre = t
        nex = t + 1

        # prepare KD tree for fast partner lookup
        nex_cells = cells_by_t[nex]
        if len(nex_cells) == 0:
            continue

        kd_data = [ cell['position'][1:] for cell in nex_cells ]
        nex_kd_tree = KDTree(kd_data)

        logger.debug(
            "Finding edges between cells in frames %d and %d (%d and %d cells)",
            pre, nex, len(cells_by_t[pre]), len(cells_by_t[nex]))

        for pre_cell in cells_by_t[pre]:

            nex_neighbor_indices = nex_kd_tree.query_ball_point(
                pre_cell['position'][1:],
                parameters.move_threshold)
            nex_neighbors = [ nex_cells[i] for i in nex_neighbor_indices ]

            for nex_cell in nex_neighbors:

                pre_center = np.array(pre_cell['position'][1:])
                nex_center = np.array(nex_cell['position'][1:])

                moved = (pre_center - nex_center)
                distance = np.linalg.norm(moved)

                # Get score from nex to pre (backwards in time).
                #
                # Cut out parent vectors around mask_roi centered at nex_center.
                # We set the fill_value to 1000 to make sure that out-of-bounds
                # voxels don't contribute to the target counts (they will point
                # very far away).

                # get the mask ROI around the next cell in 3D
                nex_roi_3d = mask_roi + daisy.Coordinate(nex_center)
                # add the time dimension
                nex_roi_4d = daisy.Roi(
                    (nex,) + nex_roi_3d.get_begin(),
                    (1,) + nex_roi_3d.get_shape())
                nex_parent_vectors = parent_vectors.fill(
                    nex_roi_4d,
                    fill_value=1000)

                # get smoothed target counts at pre
                assert nex_parent_vectors.shape[1] == 1
                counts = target_counts(
                    nex_parent_vectors.data[:,0,:],
                    nex_parent_vectors.voxel_size[1:],
                    mask.data)
                assert len(counts.shape) == 3
                counts = gaussian_filter(
                    counts,
                    parameters.sigma,
                    mode='constant')
                counts = daisy.Array(
                    counts,
                    nex_roi_3d,
                    voxel_size_3d)
                score = counts[counts.roi.get_center()]

                edges.append({
                    'source': int(nex_cell['id']),
                    'target': int(pre_cell['id']),
                    'score': float(score),
                    'distance': float(distance)
                })

    return edges
