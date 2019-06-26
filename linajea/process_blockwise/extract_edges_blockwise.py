from __future__ import absolute_import
from scipy.spatial import KDTree
import daisy
import json
import linajea
import logging
import numpy as np
import os
import time

logger = logging.getLogger(__name__)


def extract_edges_blockwise(
        db_host,
        db_name,
        sample,
        edge_move_threshold,
        block_size,
        num_workers,
        frames=None,
        frame_context=1,
        **kwargs):

    data_dir = '../01_data'

    # get absolute paths
    sample_dir = os.path.abspath(os.path.join(data_dir, sample))

    # get ROI of source
    with open(os.path.join(sample_dir, 'attributes.json'), 'r') as f:
        attributes = json.load(f)

    voxel_size = daisy.Coordinate(attributes['resolution'])
    shape = daisy.Coordinate(attributes['shape'])
    offset = daisy.Coordinate(attributes['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)

    # limit to specific frames, if given
    if frames:
        begin, end = frames
        begin -= frame_context
        end += frame_context
        crop_roi = daisy.Roi(
            (begin, None, None, None),
            (end - begin, None, None, None))
        source_roi = source_roi.intersect(crop_roi)

    # shapes in voxels
    block_write_roi = daisy.Roi(
        (0,)*4,
        daisy.Coordinate(block_size))

    pos_context = daisy.Coordinate((0,) + (edge_move_threshold,)*3)
    neg_context = daisy.Coordinate((1,) + (edge_move_threshold,)*3)
    logger.debug("Set neg context to %s", neg_context)

    input_roi = source_roi.grow(neg_context, pos_context)
    block_read_roi = block_write_roi.grow(neg_context, pos_context)

    print("Following ROIs in world units:")
    print("Input ROI       = %s" % input_roi)
    print("Block read  ROI = %s" % block_read_roi)
    print("Block write ROI = %s" % block_write_roi)
    print("Output ROI      = %s" % source_roi)

    print("Starting block-wise processing...")

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            db_name,
            db_host,
            edge_move_threshold,
            b),
        check_function=lambda b: linajea.check_function(
            b,
            'extract_edges',
            db_name,
            db_host),
        num_workers=num_workers,
        processes=True,
        read_write_conflict=False,
        fit='shrink')


def extract_edges_in_block(
        db_name,
        db_host,
        edge_move_threshold,
        block):

    logger.info(
        "Finding edges in %s, reading from %s",
        block.write_roi, block.read_roi)

    start = time.time()

    graph_provider = linajea.CandidateDatabase(
        db_name,
        db_host,
        mode='r+')
    graph = graph_provider[block.read_roi]

    if graph.number_of_nodes() == 0:
        logger.info("No cells in roi %s. Skipping", block.read_roi)
        linajea.write_done(block, 'extract_edges', db_name, db_host)
        return

    logger.info(
        "Read %d cells in %.3fs",
        graph.number_of_nodes(),
        time.time() - start)

    start = time.time()

    t_begin = block.write_roi.get_begin()[0]
    t_end = block.write_roi.get_end()[0]

    try:
        cells_by_t = {
            t: [
                (
                    cell,
                    np.array([data[d] for d in ['z', 'y', 'x']]),
                    np.array(data['parent_vector'])
                )
                for cell, data in graph.nodes(data=True)
                if 't' in data and data['t'] == t
            ]
            for t in range(t_begin - 1, t_end)
        }
    except:
        for cell, data in graph.nodes(data=True):
            print(cell, data)
        raise

    for t in range(t_begin, t_end):

        pre = t - 1
        nex = t

        logger.debug(
            "Finding edges between cells in frames %d and %d "
            "(%d and %d cells)",
            pre, nex, len(cells_by_t[pre]), len(cells_by_t[nex]))

        if len(cells_by_t[pre]) == 0 or len(cells_by_t[nex]) == 0:

            logger.debug("There are no edges between these frames, skipping")
            continue

        # prepare KD tree for fast lookup of 'pre' cells
        logger.debug("Preparing KD tree...")
        all_pre_cells = cells_by_t[pre]
        kd_data = [cell[1] for cell in all_pre_cells]
        pre_kd_tree = KDTree(kd_data)

        for i, nex_cell in enumerate(cells_by_t[nex]):

            nex_cell_id = nex_cell[0]
            nex_cell_center = nex_cell[1]
            nex_parent_center = nex_cell_center + nex_cell[2]

            pre_cells_indices = pre_kd_tree.query_ball_point(
                nex_cell_center,
                edge_move_threshold)
            pre_cells = [all_pre_cells[i] for i in pre_cells_indices]

            logger.debug(
                "Linking to %d cells in previous frame",
                len(pre_cells))

            if len(pre_cells) == 0:
                continue

            for pre_cell in pre_cells:

                pre_cell_id = pre_cell[0]
                pre_cell_center = pre_cell[1]

                moved = (pre_cell_center - nex_cell_center)
                distance = np.linalg.norm(moved)

                prediction_offset = (pre_cell_center - nex_parent_center)
                prediction_distance = np.linalg.norm(prediction_offset)

                graph.add_edge(
                    nex_cell_id, pre_cell_id,
                    distance=distance,
                    prediction_distance=prediction_distance)

    logger.info("Found %d edges", graph.number_of_edges())

    logger.info(
        "Extracted edges in %.3fs",
        time.time() - start)

    start = time.time()

    graph.write_edges(block.write_roi)

    logger.info(
        "Wrote edges in %.3fs",
        time.time() - start)
    linajea.write_done(block, 'extract_edges', db_name, db_host)
