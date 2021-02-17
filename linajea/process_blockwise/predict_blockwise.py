from __future__ import absolute_import
import json
import logging
import os
import time
import numpy as np

import daisy
from funlib.run import run

from .daisy_check_functions import check_function
from ..datasets import get_source_roi

logger = logging.getLogger(__name__)


def predict_blockwise(
        linajea_config, validate=False):
    # if validate is true, read validation data from config, else read test
    setup_dir = linajea_config.general.setup_dir
    if validate:
        data = linajea_config.validate.data
        database = linajea_config.validate.database
        checkpoint = linajea_config.validate.checkpoint
        # TODO: What if there are multiple data sources?
    else:
        data = linajea_config.test.data
        database = linajea_config.test.database
        checkpoint = linajea_config.test.checkpoint
    voxel_size = data.voxel_size
    predict_roi = data.roi
    if voxel_size is None:
        # TODO: get from data zarr/n5, or do in post_init of config
        pass
    # get context and total input and output ROI
    with open(os.path.join(setup_dir, 'test_net_config.json'), 'r') as f:
        net_config = json.load(f)
    net_input_size = net_config['input_shape']
    net_output_size = net_config['output_shape_2']
    net_input_size = daisy.Coordinate(net_input_size)*voxel_size
    net_output_size = daisy.Coordinate(net_output_size)*voxel_size
    context = (net_input_size - net_output_size)/2

    # expand predict roi to multiple of block write_roi
    predict_roi = predict_roi.snap_to_grid(net_output_size, mode='grow')

    input_roi = predict_roi.grow(context, context)
    output_roi = predict_roi

    # prepare output zarr, if necessary
    if linajea_config.predict.write_to_zarr:
        output_zarr = construct_zarr_filename(linajea_config)
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
    if linajea_config.predict.write_to_db:
        daisy.run_blockwise(
            input_roi,
            block_read_roi,
            block_write_roi,
            process_function=lambda: predict_worker(
                linajea_config.general.path,
                checkpoint,
                data.filename),
            check_function=lambda b: check_function(
                b,
                'predict',
                database.db_name,
                linajea_config.general.db_host),
            num_workers=linajea_config.predict.job.num_workers,
            read_write_conflict=False,
            max_retries=0,
            fit='valid')
    else:
        daisy.run_blockwise(
            input_roi,
            block_read_roi,
            block_write_roi,
            process_function=lambda: predict_worker(
                linajea_config.general.path,
                checkpoint,
                data.filename),
            num_workers=linajea_config.predict.job.num_workers,
            read_write_conflict=False,
            max_retries=0,
            fit='valid')


def predict_worker(linajea_config, checkpoint, datafile):

    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()
    job = linajea_config.predict.job
    if job.singularity_image is not None:
        image_path = '/nrs/funke/singularity/'
        image = image_path + job.singularity_image + '.img'
        logger.debug("Using singularity image %s" % image)
    else:
        image = None
    cmd = run(
            command='python -u %s --config %s --iteration %d --sample %s' % (
                linajea_config.predict.path_to_script,
                linajea_config.general.path,
                checkpoint,
                datafile),
            queue=job.queue,
            num_gpus=1,
            num_cpus=5,
            singularity_image=image,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            flags=['-P ' + job.lab]
            )
    logger.info("Starting predict worker...")
    logger.info("Command: %s" % str(cmd))
    daisy.call(
        cmd,
        log_out='logs/predict_%s_%d_%d.out' % (setup, worker_time, worker_id),
        log_err='logs/predict_%s_%d_%d.err' % (setup, worker_time, worker_id))

    logger.info("Predict worker finished")
