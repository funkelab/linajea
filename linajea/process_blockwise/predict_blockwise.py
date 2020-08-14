from __future__ import absolute_import
from copy import deepcopy
import datetime
import json
import logging
import os
import time

import numpy as np

import daisy
from .daisy_check_functions import check_function
from funlib.run import run

from linajea import (load_config,
                     checkOrCreateDB,
                     construct_zarr_filename)

logger = logging.getLogger(__name__)


def predict_blockwise(**kwargs):
    if kwargs.get('validation'):
        samples = kwargs['data']['val_data_dirs']
    else:
        samples = kwargs['data']['test_data_dirs']

    for sample in samples:
        sample_kwargs = deepcopy(kwargs)
        predict_blockwise_sample(sample=sample, **sample_kwargs)

def predict_blockwise_sample(**kwargs):
    if 'db_name' not in kwargs['general']:
        kwargs['general']['db_name'] = checkOrCreateDB(**kwargs)

    # get ROI of source
    data_config = load_config(os.path.join(kwargs['sample'],
                                           "data_config.toml"))
    voxel_size = daisy.Coordinate(kwargs['data']['voxel_size'])
    shape = daisy.Coordinate(data_config['general']['shape'])
    offset = daisy.Coordinate(data_config['general']['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)
    predict_roi = source_roi

    # limit to specific frames, if given
    if 'limit_to_roi_offset' in kwargs['prediction'] or \
       'frames' in kwargs['data']:
        if 'frames' in kwargs['data']:
            frames = kwargs['data']['frames']
            logger.info("Limiting prediction to frames %s" % str(frames))
            begin, end = frames
            frames_roi = daisy.Roi(
                    (begin, None, None, None),
                    (end - begin, None, None, None))
            predict_roi = predict_roi.intersect(frames_roi)
        if 'limit_to_roi_offset' in kwargs['prediction']:
            assert 'limit_to_roi_shape' in kwargs['prediction'],\
                    "Must specify shape and offset in config file"
            limit_to_roi = daisy.Roi(
                daisy.Coordinate(kwargs['prediction']['limit_to_roi_offset']),
                daisy.Coordinate(kwargs['prediction']['limit_to_roi_shape']))
            predict_roi = predict_roi.intersect(limit_to_roi)

        if 'limit_to_roi_hard' not in kwargs['prediction'] or \
           not kwargs['prediction']['limit_to_roi_hard']:
            predict_roi = predict_roi.grow(
                daisy.Coordinate(kwargs['ilp']['solve_context']),
                daisy.Coordinate(kwargs['ilp']['solve_context']))
        # predict_roi = target_roi.intersect(source_roi)

    # get context and total input and output ROI
    net_config = load_config(os.path.join(kwargs['general']['setup_dir'],
                                          'test_net_config.json'))
    net_input_size = net_config['input_shape']
    net_output_size = net_config['output_shape_2']
    net_input_size = daisy.Coordinate(net_input_size)*voxel_size
    net_output_size = daisy.Coordinate(net_output_size)*voxel_size
    context = (net_input_size - net_output_size)/2
    input_roi = predict_roi.grow(context, context)
    output_roi = predict_roi

    assert kwargs['prediction']['write_zarr'] or kwargs['prediction']['write_cells_db'], \
        "results not written! (neither zarr nor db)"

    # # prepare output zarr, if necessary
    output_path = construct_zarr_filename(**kwargs)
    if kwargs['prediction']['write_zarr']:
        parent_vectors_ds = 'volumes/parent_vectors'
        cell_indicator_ds = 'volumes/cell_indicator'
        maxima_ds = 'volumes/maxima'
        logger.debug("Preparing zarr at %s", output_path)

        daisy.prepare_ds(
                output_path,
                parent_vectors_ds,
                source_roi,
                voxel_size,
                dtype=np.float32,
                write_size=net_output_size,
                num_channels=3)
        daisy.prepare_ds(
                output_path,
                cell_indicator_ds,
                source_roi,
                voxel_size,
                dtype=np.float32,
                write_size=net_output_size,
                num_channels=1)
        daisy.prepare_ds(
                output_path,
                maxima_ds,
                source_roi,
                voxel_size,
                dtype=np.uint8,
                write_size=net_output_size,
                num_channels=1)

    # create read and write ROI
    block_write_roi = daisy.Roi((0, 0, 0, 0), net_output_size)
    block_read_roi = block_write_roi.grow(context, context)

    if not kwargs['prediction']['write_zarr'] and \
       kwargs['prediction']['write_cells_db'] and \
       os.path.exists(construct_zarr_filename(**kwargs)):
        output_roi = predict_roi.intersect(source_roi)
        input_roi = output_roi
        block_write_roi = block_write_roi.intersect(source_roi)
        block_read_roi = block_write_roi


    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s" % input_roi)
    logger.info("Block read  ROI = %s" % block_read_roi)
    logger.info("Block write ROI = %s" % block_write_roi)
    logger.info("Output ROI      = %s" % output_roi)

    logger.info("Starting block-wise processing...")
    logger.info("database: %s", kwargs['general']['db_name'])
    logger.info("sample: %s", kwargs['sample'])

    cf = []
    if kwargs['prediction']['write_zarr']:
        cf.append(lambda b: check_function(
            b,
            'predict_zarr',
            kwargs['general']['db_name'],
            kwargs['general']['db_host']))
    if kwargs['prediction']['write_cells_db']:
        cf.append(lambda b: check_function(
            b,
            'predict_cells',
            kwargs['general']['db_name'],
            kwargs['general']['db_host']))

    # process block-wise
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(**kwargs),
        check_function=lambda b: all([f(b) for f in cf]),
        num_workers=kwargs['prediction']['num_workers'],
        read_write_conflict=False,
        max_retries=0,
        fit='overhang')


def predict_worker(**kwargs):
    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()

    if 'singularity_image' in kwargs['general']:
        singularity_image = kwargs['general']['singularity_image']
        image_path = '/nrs/funke/singularity/'
        image = os.path.join(image_path, singularity_image + '.img')
        logger.debug("Using singularity image %s" % image)
    else:
        image = None
        logger.debug("Not using singularity image")

    script_name = 'predict_celegans.py'
    if not kwargs['prediction']['write_zarr'] and \
       kwargs['prediction']['write_cells_db'] and \
       os.path.exists(construct_zarr_filename(**kwargs)):
        script_name = 'write_cells_celegans.py'
    cmd = run(
            command='python %s --config %s --sample %s --iteration %d --setup_dir %s --db %s' % (
                os.path.join(kwargs['source_dir'], script_name),
                kwargs['config_file'],
                kwargs['sample'],
                kwargs['prediction']['iteration'],
                kwargs['general']['setup_dir'],
                kwargs['general']['db_name']),
            queue=kwargs['general']['queue'],
            host=kwargs['general'].get("host"),
            num_gpus=1,
            num_cpus=4,
            singularity_image=image,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            )
    logger.info("Starting predict worker...")
    logger.info("Command: %s" % str(cmd))
    daisy.call(
        cmd,
        log_out='logs/predict_%s_%d_%d.out' % (
            os.path.basename(os.path.dirname(kwargs['general']['setup_dir'])),
            worker_time, worker_id),
        log_err='logs/predict_%s_%d_%d.err' % (
            os.path.basename(os.path.dirname(kwargs['general']['setup_dir'])),
            worker_time, worker_id))

    logger.info("Predict worker finished")
