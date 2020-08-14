from __future__ import absolute_import
from copy import deepcopy
import json
import logging
import os
import time

import daisy
from .daisy_check_functions import (
        check_function, write_done,
        check_function_all_blocks, write_done_all_blocks)
from linajea import CandidateDatabase
from linajea.tracking import TrackingParameters, track
from linajea import (adjust_postprocess_roi,
                     checkOrCreateDB,
                     load_config)

logger = logging.getLogger(__name__)


def solve_blockwise(config, validation=False):
    if validation:
        samples = config['data']['val_data_dirs']
    else:
        samples = config['data']['test_data_dirs']

    for sample in samples:
        sample_config = deepcopy(config)
        solve_blockwise_sample(sample_config, sample)

def solve_blockwise_sample(config, sample):
    if 'db_name' not in config['general']:
        config['general']['db_name'] = checkOrCreateDB(config, sample)

    parameters = TrackingParameters(**config['solve'])
    block_size = daisy.Coordinate(parameters.block_size)
    context = daisy.Coordinate(parameters.context)

    # get ROI of source
    data_config = load_config(os.path.join(sample, "data_config.toml"))
    voxel_size = daisy.Coordinate(config['data']['voxel_size'])
    shape = daisy.Coordinate(data_config['general']['shape'])
    offset = daisy.Coordinate(data_config['general']['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)

    # determine parameters id from database
    graph_provider = CandidateDatabase(
        config['general']['db_name'],
        config['general']['db_host'])
    parameters_id = graph_provider.get_parameters_id(parameters)

    logger.info("from scratch %s", config['solve']['from_scratch'])
    if config['solve']['from_scratch']:
        graph_provider.set_parameters_id(parameters_id)
        graph_provider.reset_selection()

    # limit to specific frames/roi, if given
    source_roi = adjust_postprocess_roi(config, source_roi)
    logger.info("Limiting solving to roi %s", source_roi)

    block_write_roi = daisy.Roi(
        (0, 0, 0, 0),
        block_size)
    block_read_roi = block_write_roi.grow(
        context,
        context)

    step_name = 'solve_' + str(parameters_id)
    if check_function_all_blocks(step_name,
                                 config['general']['db_name'],
                                 config['general']['db_host']):
        logger.info("Step %s is already completed. Exiting", step_name)
        return True

    success = daisy.run_blockwise(
        source_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: solve_in_block(
            config,
            b,
            parameters,
            parameters_id,
            solution_roi=source_roi,
            ),
        check_function=lambda b: check_function(
            b,
            step_name,
            config['general']['db_name'],
            config['general']['db_host']),
        num_workers=config['solve']['num_workers'],
        fit='overhang')
    if success:
        write_done_all_blocks(
            step_name,
            config['general']['db_name'],
            config['general']['db_host'])
    logger.info("Finished solving, parameters id is %s", parameters_id)
    return success


def solve_in_block(config,
                   block,
                   parameters,
                   parameters_id,
                   solution_roi=None,
                   ):
    # Solution_roi is the total roi that you want a solution in
    # Limiting the block to the solution_roi allows you to solve
    # all the way to the edge, without worrying about reading
    # data from outside the solution roi
    # or paying the appear or disappear costs unnecessarily

    logger.info("Solving in block %s", block)
    if solution_roi:
        # Limit block to source_roi
        read_roi = block.read_roi.intersect(solution_roi)
        write_roi = block.write_roi.intersect(solution_roi)
    else:
        read_roi = block.read_roi
        write_roi = block.write_roi

    graph_provider = CandidateDatabase(
        config['general']['db_name'],
        config['general']['db_host'],
        mode='r+',
        parameters_id=parameters_id)
    start_time = time.time()
    nodes_filter = None
    try:
        if parameters.masked_nodes is not None:
            nodes_filter = {parameters.masked_nodes: True}
    except:
        pass
    graph = graph_provider.get_graph(
        read_roi,
        nodes_filter=nodes_filter,
        edge_attrs=["prediction_distance",
                    "distance",
                    graph_provider.selected_key]
    )

    # remove dangling nodes and edges
    dangling_nodes = [
        n
        for n, data in graph.nodes(data=True)
        if 't' not in data
    ]
    graph.remove_nodes_from(dangling_nodes)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    logger.info("Reading graph with %d nodes and %d edges took %s seconds",
                num_nodes, num_edges, time.time() - start_time)

    if num_edges == 0:
        logger.info("No edges in roi %s. Skipping",
                    read_roi)
        write_done(block, 'solve_' + str(parameters_id),
                   config['general']['db_name'],
                   config['general']['db_host'])
        return 0

    frames = [read_roi.get_offset()[0],
              read_roi.get_offset()[0] + read_roi.get_shape()[0]]
    track(graph, parameters, graph_provider.selected_key, frames=frames)
    start_time = time.time()
    graph.update_edge_attrs(
            write_roi,
            attributes=[graph_provider.selected_key])
    logger.info("Updating attribute %s for %d edges took %s seconds",
                graph_provider.selected_key,
                num_edges,
                time.time() - start_time)
    write_done(block, 'solve_' + str(parameters_id),
               config['general']['db_name'],
               config['general']['db_host'])
    return 0
