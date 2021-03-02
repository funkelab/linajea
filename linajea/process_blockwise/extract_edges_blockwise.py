from __future__ import absolute_import
import logging
import time

from scipy.spatial import cKDTree
import numpy as np

import daisy
import linajea
from .daisy_check_functions import write_done, check_function

logger = logging.getLogger(__name__)


def extract_edges_blockwise(linajea_config):

    data = linajea_config.inference.data_source
    voxel_size = daisy.Coordinate(data.voxel_size)
    extract_roi = daisy.Roi(offset=data.roi.offset,
                            shape=data.roi.shape)
    # allow for solve context
    extract_roi = extract_roi.grow(
            daisy.Coordinate(linajea_config.solve.parameters[0].context),
            daisy.Coordinate(linajea_config.solve.parameters[0].context))
    # but limit to actual file roi
    extract_roi = extract_roi.intersect(
        daisy.Roi(offset=data.datafile.file_roi.offset,
                  shape=data.datafile.file_roi.shape))

    # block size in world units
    block_write_roi = daisy.Roi(
        (0,)*4,
        daisy.Coordinate(linajea_config.extract.block_size))

    max_edge_move_th = max(linajea_config.extract.edge_move_threshold.values())
    pos_context = daisy.Coordinate((0,) + (max_edge_move_th,)*3)
    neg_context = daisy.Coordinate((1,) + (max_edge_move_th,)*3)
    logger.debug("Set neg context to %s", neg_context)

    input_roi = extract_roi.grow(neg_context, pos_context)
    block_read_roi = block_write_roi.grow(neg_context, pos_context)

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s", input_roi)
    logger.info("Block read  ROI = %s", block_read_roi)
    logger.info("Block write ROI = %s", block_write_roi)
    logger.info("Output ROI      = %s", extract_roi)

    logger.info("Starting block-wise processing...")
    logger.info("Sample: %s", data.datafile.filename)
    logger.info("DB: %s", data.db_name)

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            linajea_config,
            b),
        check_function=lambda b: check_function(
            b,
            'extract_edges',
            data.db_name,
            linajea_config.general.db_host),
        num_workers=linajea_config.extract.job.num_workers,
        processes=True,
        read_write_conflict=False,
        fit='overhang')


def extract_edges_in_block(
        linajea_config,
        block):

    logger.info(
        "Finding edges in %s, reading from %s",
        block.write_roi, block.read_roi)

    data = linajea_config.inference.data_source

    start = time.time()

    graph_provider = linajea.CandidateDatabase(
        data.db_name,
        linajea_config.general.db_host,
        mode='r+')
    graph = graph_provider[block.read_roi]

    if graph.number_of_nodes() == 0:
        logger.info("No cells in roi %s. Skipping", block.read_roi)
        write_done(block, 'extract_edges', data.db_name,
                   linajea_config.general.db_host)
        return 0

    logger.info(
        "Read %d cells in %.3fs",
        graph.number_of_nodes(),
        time.time() - start)

    start = time.time()

    t_begin = block.write_roi.get_begin()[0]
    t_end = block.write_roi.get_end()[0]

    cells_by_t = {
        t: [
            (
                cell,
                np.array([attrs[d] for d in ['z', 'y', 'x']]),
                np.array(attrs['parent_vector'])
            )
            for cell, attrs in graph.nodes(data=True)
            if 't' in attrs and attrs['t'] == t
        ]
        for t in range(t_begin - 1, t_end)
    }

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
        pre_kd_tree = cKDTree(kd_data)

        for th, val in linajea_config.extract.edge_move_threshold.items():
            if th == -1 or t < int(th):
                edge_move_threshold = val
                break

        for i, nex_cell in enumerate(cells_by_t[nex]):

            nex_cell_id = nex_cell[0]
            nex_cell_center = nex_cell[1]
            nex_parent_center = nex_cell_center + nex_cell[2]

            if use_pv_distance:
                pre_cells_indices = pre_kd_tree.query_ball_point(
                    nex_parent_center,
                    edge_move_threshold)

            else:
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
    write_done(block, 'extract_edges', data.db_name,
               linajea_config.general.db_host)
    return 0
