from __future__ import absolute_import
from copy import deepcopy
import json
import logging
import os
import time

import numpy as np
from scipy.spatial import KDTree

import daisy
from .daisy_check_functions import write_done, check_function
import linajea
from linajea import (adjust_postprocess_roi,
                     checkOrCreateDB,
                     load_config)

logger = logging.getLogger(__name__)


def extract_edges_blockwise(config, validation=False):
    if validation:
        samples = config['data']['val_data_dirs']
    else:
        samples = config['data']['test_data_dirs']

    for sample in samples:
        sample_config = deepcopy(config)
        extract_edges_blockwise_sample(sample_config, sample)

def extract_edges_blockwise_sample(config, sample):
    if 'db_name' not in config['general']:
        config['general']['db_name'] = checkOrCreateDB(config, sample)

    # get ROI of source
    data_config = load_config(os.path.join(sample, "data_config.toml"))
    voxel_size = daisy.Coordinate(config['data']['voxel_size'])
    shape = daisy.Coordinate(data_config['general']['shape'])
    offset = daisy.Coordinate(data_config['general']['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)

    # limit to specific frames/roi, if given
    source_roi = adjust_postprocess_roi(config, source_roi, use_context=True)
    logger.info("Limiting extract edges to roi {}".format(source_roi))

    # shapes in voxels
    block_write_roi = daisy.Roi(
        (0,)*4,
        daisy.Coordinate(config['extract_edges']['block_size']))

    edge_move_threshold = config['extract_edges']['edge_move_threshold']
    if isinstance(edge_move_threshold, dict):
        max_th = max(edge_move_threshold.values())
    else:
        max_th = edge_move_threshold
    pos_context = daisy.Coordinate((0,) + (max_th,)*3)
    neg_context = daisy.Coordinate((1,) + (max_th,)*3)
    logger.debug("Set neg context to %s", neg_context)

    input_roi = source_roi.grow(neg_context, pos_context)
    block_read_roi = block_write_roi.grow(neg_context, pos_context)

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s" % input_roi)
    logger.info("Block read  ROI = %s" % block_read_roi)
    logger.info("Block write ROI = %s" % block_write_roi)
    logger.info("Output ROI      = %s" % source_roi)

    logger.info("Starting block-wise processing...")
    logger.info("database: %s", config['general']['db_name'])
    logger.info("sample: %s", sample)

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(config, b),
        check_function=lambda b: check_function(
            b,
            'extract_edges',
            config['general']['db_name'],
            config['general']['db_host']),
        num_workers=config['extract_edges']['num_workers'],
        processes=True,
        read_write_conflict=False,
        fit='overhang')
        # fit='shrink')


def extract_edges_in_block(config, block):
    logger.info(
        "Finding edges in %s, reading from %s",
        block.write_roi, block.read_roi)

    start = time.time()

    graph_provider = linajea.CandidateDatabase(
        config['general']['db_name'],
        config['general']['db_host'],
        mode='r+')
    graph = graph_provider[block.read_roi]

    if graph.number_of_nodes() == 0:
        logger.info("No cells in roi %s. Skipping", block.read_roi)
        write_done(block, 'extract_edges',
                   config['general']['db_name'],
                   config['general']['db_host'])
        return 0

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

        edge_move_threshold = config['extract_edges']['edge_move_threshold']
        for i, nex_cell in enumerate(cells_by_t[nex]):

            nex_cell_id = nex_cell[0]
            nex_cell_center = nex_cell[1]
            nex_parent_center = nex_cell_center + nex_cell[2]

            if isinstance(edge_move_threshold, dict):
                for th, val in edge_move_threshold.items():
                    if t < int(th):
                        emt = val
                        break
            else:
                emt = edge_move_threshold
            pre_cells_indices = pre_kd_tree.query_ball_point(
                nex_cell_center,
                emt)
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
    write_done(block, 'extract_edges',
               config['general']['db_name'],
               config['general']['db_host'])
    return 0
