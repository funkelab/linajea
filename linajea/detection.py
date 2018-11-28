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

        cell_score_threshold (optional ``int``)

            Only consider cells with scores above this threshold.

    '''

    def __init__(self, move_threshold, pool_radius, sigma, cell_score_threshold=None):

        self.move_threshold = move_threshold
        self.pool_radius = pool_radius
        self.sigma = sigma
        self.cell_score_threshold = cell_score_threshold

def find_cells(
        target_counts,
        parameters):

    if parameters.downsample:

        downsample = tuple(parameters.downsample)

        logger.debug("Downsampling target_counts...")
        start = time.time()
        downsampled = block_reduce(target_counts.to_ndarray(), (1,) + downsample, np.sum)
        voxel_size = target_counts.voxel_size*daisy.Coordinate((1,) + downsample)
        target_counts = daisy.Array(
            downsampled,
            daisy.Roi(
                target_counts.roi.get_begin(),
                voxel_size*downsampled.shape),
            voxel_size)
        logger.debug("%.3fs", time.time() - start)
        logger.debug("new voxel size of target_counts: %s"%(target_counts.voxel_size,))

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
            if cell['position'][0] == t and (not parameters.cell_score_threshold or
                                             cell['score'] >= parameters.cell_score_threshold)
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
    mask_roi -= (shape/2)*voxel_size_3d
    mask = daisy.Array(
        sphere(radius_vx).astype(np.int32),
        mask_roi,
        voxel_size_3d)

    edges = []
    for t in range(t_begin, t_end - 1):

        pre = t
        nex = t + 1

        logger.debug(
            "Finding edges between cells in frames %d and %d (%d and %d cells)",
            pre, nex, len(cells_by_t[pre]), len(cells_by_t[nex]))

        if len(cells_by_t[pre]) == 0 or len(cells_by_t[nex]) == 0:

            logger.debug("There are no edges between these frames, skipping")
            continue

        # Compute edge scores backwards in time. For each 'nex' cell, get the
        # masked target counts in the previous frame (masked meaning we consider
        # only parent vectors in a ball around 'nex'). This is done in a ROI
        # around the 'nex' and 'pre' cells only. The edge score to all 'pre'
        # cells can then directly be read out from the target counts for this
        # ROI.
        #
        # 1. get a 'nex' cell
        # 2. find all 'pre' cells within 'move_threshold'
        # 3. get ROI of all 'pre' cells
        # 4. grow ROI by 2*'sigma' (this ensures we still count parent vectors
        #    that don't point right to the center of the 'pre' cells)
        # 5. get parent vectors in ROI in 'nex' frame
        # 6. create a mask for same ROI around 'nex' cell
        # 7. compute target counts for that ROI
        # 8. smooth target counts with 'sigma'
        # 9. for each 'pre' cell, read out the edge score from the smoothed
        #    target counts

        # prepare KD tree for fast lookup of 'pre' cells
        logger.debug("Preparing KD tree...")
        all_pre_cells = cells_by_t[pre]
        kd_data = [ cell['position'][1:] for cell in all_pre_cells ]
        pre_kd_tree = KDTree(kd_data)

        read_time = 0
        target_counts_time = 0
        smooth_time = 0

        total_time_start = time.time()

        # 1. get a 'nex' cell
        for i, nex_cell in enumerate(cells_by_t[nex]):

            if i > 0 and i % 100 == 0:
                logger.info("Avg. read time        : %.3fs", float(read_time)/i)
                logger.info("Avg. target count time: %.3fs", float(target_counts_time)/i)
                logger.info("Avg. smooth time      : %.3fs", float(smooth_time)/i)
                logger.info("Avg. total time       : %.3fs", float(time.time() - total_time_start)/i)

            nex_cell_center = daisy.Coordinate(nex_cell['position'][1:])
            nex_mask_roi = mask.roi + nex_cell_center
            nex_mask_roi = nex_mask_roi.snap_to_grid(voxel_size_3d, mode='closest')

            logger.debug(
                "Processing cell %d at %s, masking in ROI %s",
                nex_cell['id'], nex_cell_center, nex_mask_roi)

            # 2. find all 'pre' cells within 'move_threshold'

            pre_cells_indices = pre_kd_tree.query_ball_point(
                nex_cell_center,
                parameters.move_threshold)
            pre_cells = [ all_pre_cells[i] for i in pre_cells_indices ]

            logger.debug(
                "Linking to %d cells in previous frame",
                len(pre_cells))

            if len(pre_cells) == 0:
                continue

            # 3. get ROI of all 'pre' cells

            roi_3d = roi_from_points([
                pre_cell['position'][1:]
                for pre_cell in pre_cells
            ])
            roi_3d = roi_3d.union(nex_mask_roi)

            # 4. grow ROI by 2*'sigma'

            context = daisy.Coordinate((math.ceil(2*s) for s in parameters.sigma))
            roi_3d = roi_3d.grow(context, context)
            roi_3d = roi_3d.snap_to_grid(voxel_size_3d, mode='grow')

            logger.debug("Context ROI: %s", roi_3d)

            # 5. get parent vectors in ROI in 'nex' frame

            pre_roi = daisy.Roi(
                (pre,) + roi_3d.get_begin(),
                (1,) + roi_3d.get_shape())
            nex_roi = daisy.Roi(
                (nex,) + roi_3d.get_begin(),
                (1,) + roi_3d.get_shape())

            start = time.time()
            nex_parent_vectors = parent_vectors.to_ndarray(
                nex_roi,
                fill_value=1000)
            assert nex_parent_vectors.shape[1] == 1
            read_time += time.time() - start

            # 6. create a mask for same ROI around 'nex' cell

            nex_mask = daisy.Array(
                np.zeros(nex_parent_vectors.shape[-3:], dtype=np.int32),
                roi_3d,
                voxel_size_3d)
            nex_mask[nex_mask_roi] = mask

            # 7. compute target counts for that ROI

            start = time.time()
            counts = daisy.Array(
                target_counts(
                    nex_parent_vectors[:,0,:],
                    voxel_size_3d,
                    mask=nex_mask.to_ndarray()),
                roi_3d,
                voxel_size_3d)
            target_counts_time += time.time() - start

            # 8. smooth target counts with 'sigma'

            start = time.time()
            counts.data = gaussian_filter(
                counts.data.astype(np.float),
                parameters.sigma,
                mode='constant')
            smooth_time += time.time() - start

            # 9. for each 'pre' cell, read out the edge score from the smoothed
            #    target counts

            for pre_cell in pre_cells:

                pre_center = np.array(pre_cell['position'][1:])
                nex_center = np.array(nex_cell['position'][1:])

                moved = (pre_center - nex_center)
                distance = np.linalg.norm(moved)

                score = counts[daisy.Coordinate(pre_cell['position'][1:])]

                edges.append({
                    'source': int(nex_cell['id']),
                    'target': int(pre_cell['id']),
                    'score': float(score),
                    'distance': float(distance)
                })

    return edges

def roi_from_points(points):

    begin = daisy.Coordinate((min(c) for c in zip(*points)))
    end = daisy.Coordinate((max(c) + 1 for c in zip(*points)))

    return daisy.Roi(
        begin,
        end - begin)
