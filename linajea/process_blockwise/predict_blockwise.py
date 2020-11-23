from __future__ import absolute_import
import json
import logging
import os
import time
import numpy as np

import daisy
from funlib.run import run

from .daisy_check_functions import check_function
from linajea import load_config
from ..datasets import get_source_roi

logger = logging.getLogger(__name__)


def predict_blockwise(
        config_file,
        iteration
        ):
    config = {
            "solve_context": daisy.Coordinate((2, 100, 100, 100)),
            "num_workers": 16,
            "data_dir": '../01_data',
            "setups_dir": '../02_setups',
        }
    master_config = load_config(config_file)
    config.update(master_config['general'])
    config.update(master_config['predict'])
    sample = config['sample']
    data_dir = config['data_dir']
    setup = config['setup']
    # solve_context = daisy.Coordinate(master_config['solve']['context'])
    setup_dir = os.path.abspath(
            os.path.join(config['setups_dir'], setup))
    voxel_size, source_roi = get_source_roi(data_dir, sample)
    predict_roi = source_roi

    # limit to specific frames, if given
    if 'limit_to_roi_offset' in config or 'frames' in config:
        if 'frames' in config:
            frames = config['frames']
            logger.info("Limiting prediction to frames %s" % str(frames))
            begin, end = frames
            frames_roi = daisy.Roi(
                    (begin, None, None, None),
                    (end - begin, None, None, None))
            predict_roi = predict_roi.intersect(frames_roi)
        if 'limit_to_roi_offset' in config:
            assert 'limit_to_roi_shape' in config,\
                    "Must specify shape and offset in config file"
            limit_to_roi = daisy.Roi(
                    daisy.Coordinate(config['limit_to_roi_offset']),
                    daisy.Coordinate(config['limit_to_roi_shape']))
            predict_roi = predict_roi.intersect(limit_to_roi)
        # Given frames and rois are the prediction region,
        # not the solution region
        # predict_roi = target_roi.grow(solve_context, solve_context)
        # predict_roi = predict_roi.intersect(source_roi)

    # get context and total input and output ROI
    with open(os.path.join(setup_dir, 'test_net_config.json'), 'r') as f:
        net_config = json.load(f)
    net_input_size = net_config['input_shape']
    net_output_size = net_config['output_shape_2']
    net_input_size = daisy.Coordinate(net_input_size)*voxel_size
    net_output_size = daisy.Coordinate(net_output_size)*voxel_size
    context = (net_input_size - net_output_size)/2
    input_roi = predict_roi.grow(context, context)
    output_roi = predict_roi

    # prepare output zarr, if necessary
    if 'output_zarr' in config:
        output_zarr = config['output_zarr']
        parent_vectors_ds = 'volumes/parent_vectors'
        cell_indicator_ds = 'volumes/cell_indicator'
        output_path = os.path.join(setup_dir, output_zarr)
        logger.debug("Preparing zarr at %s" % output_path)
        daisy.prepare_ds(
                output_path,
                parent_vectors_ds,
                output_roi,
                voxel_size,
                dtype=np.float32,
                write_size=net_output_size,
                num_channels=3)
        daisy.prepare_ds(
                output_path,
                cell_indicator_ds,
                output_roi,
                voxel_size,
                dtype=np.float32,
                write_size=net_output_size,
                num_channels=1)

    # create read and write ROI
    block_write_roi = daisy.Roi((0, 0, 0, 0), net_output_size)
    block_read_roi = block_write_roi.grow(context, context)

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s" % input_roi)
    logger.info("Block read  ROI = %s" % block_read_roi)
    logger.info("Block write ROI = %s" % block_write_roi)
    logger.info("Output ROI      = %s" % output_roi)

    logger.info("Starting block-wise processing...")

    # process block-wise
    if 'db_name' in config:
        daisy.run_blockwise(
            input_roi,
            block_read_roi,
            block_write_roi,
            process_function=lambda: predict_worker(
                config_file,
                iteration),
            check_function=lambda b: check_function(
                b,
                'predict',
                config['db_name'],
                config['db_host']),
            num_workers=config['num_workers'],
            read_write_conflict=False,
            max_retries=0,
            fit='overhang')
    else:
        daisy.run_blockwise(
            input_roi,
            block_read_roi,
            block_write_roi,
            process_function=lambda: predict_worker(
                config_file,
                iteration),
            num_workers=config['num_workers'],
            read_write_conflict=False,
            max_retries=0,
            fit='overhang')


def predict_worker(
        config_file,
        iteration):
    config = {
            "singularity_image": 'linajea/linajea:v1.1',
            "queue": 'slowpoke',
            'setups_dir': '../02_setups'
        }
    master_config = load_config(config_file)
    config.update(master_config['general'])
    config.update(master_config['predict'])
    singularity_image = config['singularity_image']
    queue = config['queue']
    setups_dir = config['setups_dir']
    setup = config['setup']
    chargeback = config['lab']

    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()
    if singularity_image is not None:
        image_path = '/nrs/funke/singularity/'
        image = image_path + singularity_image + '.img'
        logger.debug("Using singularity image %s" % image)
    else:
        image = None
    cmd = run(
            command='python -u %s --config %s --iteration %d' % (
                os.path.join(setups_dir, 'predict.py'),
                config_file,
                iteration),
            queue=queue,
            num_gpus=1,
            num_cpus=5,
            singularity_image=image,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            flags=['-P ' + chargeback]
            )
    logger.info("Starting predict worker...")
    logger.info("Command: %s" % str(cmd))
    daisy.call(
        cmd,
        log_out='logs/predict_%s_%d_%d.out' % (setup, worker_time, worker_id),
        log_err='logs/predict_%s_%d_%d.err' % (setup, worker_time, worker_id))

    logger.info("Predict worker finished")
