from __future__ import absolute_import
import json
import logging
import os
import time

import daisy
from funlib.run import run

from .daisy_check_functions import check_function

logger = logging.getLogger(__name__)


def predict_blockwise(
        setup,
        iteration,
        sample,
        db_host,
        db_name,
        cell_score_threshold=0,
        frames=None,
        limit_to_roi=None,
        solve_context=None,
        num_workers=16,
        singularity_image='linajea/linajea:v1.1',
        queue='slowpoke',
        **kwargs):

    data_dir = '../01_data'
    setup_dir = '../02_setups'

    # get absolute paths
    if os.path.isfile(sample) or sample.endswith((".zarr", ".n5")):
        sample_dir = os.path.abspath(os.path.join(data_dir,
                                                  os.path.dirname(sample)))
    else:
        sample_dir = os.path.abspath(os.path.join(data_dir, sample))

    setup_dir = os.path.abspath(os.path.join(setup_dir, setup))
    # get ROI of source
    with open(os.path.join(sample_dir, 'attributes.json'), 'r') as f:
        attributes = json.load(f)

    voxel_size = daisy.Coordinate(attributes['resolution'])
    shape = daisy.Coordinate(attributes['shape'])
    offset = daisy.Coordinate(attributes['offset'])
    source_roi = daisy.Roi(offset, shape*voxel_size)

    # limit to specific frames, if given
    if limit_to_roi is not None or frames is not None:
        target_roi = source_roi
        if frames:
            logger.info("Limiting prediction to frames %s" % str(frames))
            begin, end = frames
            frames_roi = daisy.Roi(
                    (begin, None, None, None),
                    (end - begin, None, None, None))
            target_roi = target_roi.intersect(frames_roi)
        if limit_to_roi:
            target_roi = target_roi.intersect(limit_to_roi)
        if solve_context is None:
            solve_context = daisy.Coordinate((2, 100, 100, 100))
        predict_roi = target_roi.grow(solve_context, solve_context)
        predict_roi = predict_roi.intersect(source_roi)
    else:
        predict_roi = source_roi

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
    daisy.run_blockwise(
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: predict_worker(
            setup,
            iteration,
            sample,
            db_host,
            db_name,
            singularity_image,
            queue,
            cell_score_threshold),
        check_function=lambda b: check_function(
            b,
            'predict',
            db_name,
            db_host),
        num_workers=num_workers,
        read_write_conflict=False,
        max_retries=0)


def predict_worker(
        setup,
        iteration,
        sample,
        db_host,
        db_name,
        singularity_image,
        queue,
        cell_score_threshold):
    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()
    image_path = '/nrs/funke/singularity/'
    image = image_path + singularity_image + '.img'
    logger.debug("Using singularity image %s" % image)
    cmd = run(
            command='python -u %s %d %s %s %s %f' % (
                os.path.join('../02_setups', setup, 'predict.py'),
                iteration,
                sample,
                db_host,
                db_name,
                cell_score_threshold
                ),
            queue=queue,
            num_gpus=1,
            num_cpus=4,
            singularity_image=image,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            )
    logger.info("Starting predict worker...")

    daisy.call(
        cmd,
        log_out='logs/predict_%s_%d_%d.out' % (setup, worker_time, worker_id),
        log_err='logs/predict_%s_%d_%d.err' % (setup, worker_time, worker_id))

    logger.info("Predict worker finished")
