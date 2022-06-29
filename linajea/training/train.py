"""Script for training process

Create model and train
"""
from __future__ import print_function
import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import logging
import time
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import gunpowder as gp

from linajea.gunpowder_nodes import (TracksSource, AddMovementVectors,
                                     ShiftAugment, ShuffleChannels, Clip,
                                     NoOp, NormalizeMinMax, NormalizeMeanStd,
                                     NormalizeMedianMad,
                                     RandomLocationExcludeTime)
from linajea.config import (load_config,
                            maybe_fix_config_paths_to_machine_and_load,
                            TrackingConfig)


from . import torch_model
from . import torch_loss
from .utils import (get_latest_checkpoint,
                   Cast)


logger = logging.getLogger(__name__)


def train(config):
    """Main train function

    All information is taken from config object (what model architecture,
    optimizer, loss, data, augmentation etc to use)
    Train for config.train.max_iterations steps.
    Optionally compute interleaved validation statistics.

    Args
    ----
    config: TrackingConfig
        Tracking configuration object, has to contain at least model,
        train and data configuration, optionally augment
    """
    # Get the latest checkpoint
    checkpoint_basename = os.path.join(config.general.setup_dir, 'train_net')
    latest_checkpoint, trained_until = get_latest_checkpoint(checkpoint_basename)
    # training already done?
    if trained_until >= config.train.max_iterations:
        return

    anchor = gp.ArrayKey('ANCHOR')
    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    tracks = gp.PointsKey('TRACKS')
    center_tracks = gp.PointsKey('CENTER_TRACKS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    cell_center = gp.ArrayKey('CELL_CENTER')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    grad_cell_indicator = gp.ArrayKey('GRAD_CELL_INDICATOR')
    if not config.model.train_only_cell_indicator:
        movement_vectors = gp.ArrayKey('MOVEMENT_VECTORS')
        pred_movement_vectors = gp.ArrayKey('PRED_MOVEMENT_VECTORS')
        cell_mask = gp.ArrayKey('CELL_MASK')
        grad_movement_vectors = gp.ArrayKey('GRAD_MOVEMENT_VECTORS')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = torch_model.UnetModelWrapper(config, trained_until)
    model.init_layers()
    try:
        model = model.to(device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to move model to device. If you are using a child process "
            "to run your model, maybe you already initialized CUDA by sending "
            "your model to device in the main process."
        ) from e

    input_shape, output_shape_1, output_shape_2 = model.inout_shapes(device=device)
    logger.debug("Model: %s", model)

    voxel_size = gp.Coordinate(config.train_data.data_sources[0].voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size_1 = gp.Coordinate(output_shape_1) * voxel_size
    output_size_2 = gp.Coordinate(output_shape_2) * voxel_size
    center_size = gp.Coordinate(output_shape_2) * voxel_size
    # add a buffer in time to avoid choosing a random location based on points
    # in only one frame, because then all points are rejected as being on the
    # lower boundary of that frame
    center_size = center_size + gp.Coordinate((1, 0, 0, 0))
    logger.debug("Center size: {}".format(center_size))
    logger.debug("Output size 1: {}".format(output_size_1))
    logger.debug("Voxel size: {}".format(voxel_size))

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(tracks, output_size_1)
    request.add(center_tracks, center_size)
    request.add(cell_indicator, output_size_1)
    request.add(cell_center, output_size_1)
    request.add(anchor, output_size_2)
    request.add(raw_cropped, output_size_2)
    request.add(maxima, output_size_2)
    if not config.model.train_only_cell_indicator:
        request.add(movement_vectors, output_size_1)
        request.add(cell_mask, output_size_1)
    logger.debug("REQUEST: %s" % str(request))
    snapshot_request = gp.BatchRequest({
        raw: request[raw],
        pred_cell_indicator: request[movement_vectors],
        grad_cell_indicator: request[movement_vectors]
    })
    snapshot_request.add(pred_cell_indicator, output_size_1)
    snapshot_request.add(raw_cropped, output_size_2)
    snapshot_request.add(maxima, output_size_2)
    if not config.model.train_only_cell_indicator:
        snapshot_request.add(pred_movement_vectors, output_size_1)
        snapshot_request.add(grad_movement_vectors, output_size_1)
    logger.debug("Snapshot request: %s" % str(snapshot_request))


    train_sources = get_sources(config, raw, anchor, tracks, center_tracks,
                                config.train_data.data_sources, val=False)

    # Do interleaved validation?
    if config.train.val_log_step is not None:
        val_sources = get_sources(config, raw, anchor, tracks, center_tracks,
                                  config.validate_data.data_sources, val=True)

    # set up pipeline:
    # load data source and
    # choose augmentations depending on config
    augment = config.train.augment
    train_pipeline = (
        tuple(train_sources) +
        gp.RandomProvider() +

        (gp.ElasticAugment(
            augment.elastic.control_point_spacing,
            augment.elastic.jitter_sigma,
            [augment.elastic.rotation_min*np.pi/180.0,
             augment.elastic.rotation_max*np.pi/180.0],
            rotation_3d=augment.elastic.rotation_3d,
            subsample=augment.elastic.subsample,
            use_fast_points_transform=augment.elastic.use_fast_points_transform,
            spatial_dims=3,
            temporal_dim=True) \
         if augment.elastic is not None else NoOp()) +

        (ShiftAugment(
            prob_slip=augment.shift.prob_slip,
            prob_shift=augment.shift.prob_shift,
            sigma=augment.shift.sigma,
            shift_axis=0) \
         if augment.shift is not None else NoOp()) +

        (ShuffleChannels(raw) \
         if augment.shuffle_channels else NoOp()) +

        (gp.SimpleAugment(
            mirror_only=augment.simple.mirror,
            transpose_only=augment.simple.transpose) \
         if augment.simple is not None else NoOp()) +

        (gp.ZoomAugment(
            factor_min=augment.zoom.factor_min,
            factor_max=augment.zoom.factor_max,
            spatial_dims=augment.zoom.spatial_dims,
            order={raw: 1,
                   }) \
         if augment.zoom is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='gaussian',
            var=augment.noise_gaussian.var,
            clip=False,
            check_val_range=False) \
         if augment.noise_gaussian is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='speckle',
            var=augment.noise_speckle.var,
            clip=False,
            check_val_range=False) \
         if augment.noise_speckle is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='s&p',
            amount=augment.noise_saltpepper.amount,
            clip=False,
            check_val_range=False) \
         if augment.noise_saltpepper is not None else NoOp()) +

        (gp.HistogramAugment(
            raw,
            # raw_tmp,
            range_low=augment.histogram.range_low,
            range_high=augment.histogram.range_high,
            z_section_wise=False) \
        if augment.histogram is not None else NoOp())  +

        (gp.IntensityAugment(
            raw,
            scale_min=augment.intensity.scale[0],
            scale_max=augment.intensity.scale[1],
            shift_min=augment.intensity.shift[0],
            shift_max=augment.intensity.shift[1],
            z_section_wise=False,
            clip=False) \
         if augment.intensity is not None else NoOp()) +

        (AddMovementVectors(
            tracks,
            movement_vectors,
            cell_mask,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            radius=config.train.object_radius,
            move_radius=config.train.move_radius,
            dense=not config.general.sparse) \
         if not config.model.train_only_cell_indicator else NoOp()) +

        (gp.Reject(
            ensure_nonempty=tracks,
            mask=cell_mask,
            min_masked=0.0001,
        ) \
         if config.general.sparse else NoOp()) +

        gp.RasterizeGraph(
            tracks,
            cell_indicator,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=config.train.rasterize_radius,
                mode='peak')) +

        gp.RasterizeGraph(
            tracks,
            cell_center,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.1,) + voxel_size[1:],
                mode='point'))
        +

        (gp.PreCache(
            cache_size=config.train.cache_size,
            num_workers=config.train.job.num_workers) \
         if config.train.job.num_workers > 1 else NoOp())
    )

    # set up optional validation path without augmentations
    if config.train.val_log_step is not None:
        val_pipeline = (
            tuple(val_sources) +
            gp.RandomProvider() +

            (AddMovementVectors(
                tracks,
                movement_vectors,
                cell_mask,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                radius=config.train.object_radius,
                move_radius=config.train.move_radius,
                dense=not config.general.sparse) \
             if not config.model.train_only_cell_indicator else NoOp()) +

            (gp.Reject(
                ensure_nonempty=tracks,
                mask=cell_mask,
                min_masked=0.0001,
                reject_probability=augment.reject_empty_prob
            ) \
             if config.general.sparse else NoOp()) +

            gp.Reject(
                ensure_nonempty=center_tracks,
                # always reject emtpy batches in validation branch
                reject_probability=1.0
            ) +

            gp.RasterizePoints(
                tracks,
                cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=config.train.rasterize_radius,
                    mode='peak')) +

            gp.RasterizePoints(
                tracks,
                cell_center,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=(0.1,) + voxel_size[1:],
                    mode='point'))
            +

            (gp.PreCache(
                cache_size=config.train.cache_size,
                num_workers=1) \
             if config.train.job.num_workers > 1 else NoOp())
        )

    if config.train.val_log_step is not None:
        pipeline = (
            (train_pipeline, val_pipeline) +
            gp.TrainValProvider(step=config.train.val_log_step,
                                init_step=trained_until))
    else:
        pipeline = train_pipeline


    inputs={
        'raw': raw,
    }
    if not config.model.train_only_cell_indicator:
        inputs['cell_mask'] = cell_mask
        inputs['gt_movement_vectors'] = movement_vectors

    outputs={
        0: pred_cell_indicator,
        1: maxima,
        2: raw_cropped,
    }
    if not config.model.train_only_cell_indicator:
        outputs[3] = pred_movement_vectors

    loss_inputs={
        'gt_cell_indicator': cell_indicator,
        'cell_indicator': pred_cell_indicator,
        'maxima': maxima,
        'gt_cell_center': cell_center,
    }
    if not config.model.train_only_cell_indicator:
        loss_inputs['cell_mask'] = cell_mask
        loss_inputs['gt_movement_vectors'] = movement_vectors
        loss_inputs['movement_vectors'] = pred_movement_vectors

    gradients = {
        0: grad_cell_indicator,
    }
    if not config.model.train_only_cell_indicator:
        gradients[3] = grad_movement_vectors

    snapshot_datasets = {
        raw: 'volumes/raw',
        anchor: 'volumes/anchor',
        raw_cropped: 'volumes/raw_cropped',
        cell_indicator: 'volumes/cell_indicator',
        cell_center: 'volumes/cell_center',

        pred_cell_indicator: 'volumes/pred_cell_indicator',
        maxima: 'volumes/maxima',
        grad_cell_indicator: 'volumes/grad_cell_indicator',
    }
    if not config.model.train_only_cell_indicator:
        snapshot_datasets[cell_mask] = 'volumes/cell_mask'
        snapshot_datasets[movement_vectors] = 'volumes/movement_vectors'
        snapshot_datasets[pred_movement_vectors] = 'volumes/pred_movement_vectors'
        snapshot_datasets[grad_movement_vectors] = 'volumes/grad_movement_vectors'

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("requires_grad enabled for:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug("%s", name)

    # create optimizer
    opt = getattr(torch.optim, config.optimizerTorch.optimizer)(
        model.parameters(), **config.optimizerTorch.get_kwargs())

    # if new training, save initial state to disk
    if trained_until == 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            os.path.join(config.general.setup_dir, "train_net_checkpoint_0"))

    # create loss object
    loss = torch_loss.LossWrapper(config, current_step=trained_until)

    # and add training gunpowder node
    pipeline = (
        pipeline +
        gp.torch.Train(
            model=model,
            loss=loss,
            optimizer=opt,
            checkpoint_basename=os.path.join(config.general.setup_dir, 'train_net'),
            inputs=inputs,
            outputs=outputs,
            loss_inputs=loss_inputs,
            gradients=gradients,
            log_dir=os.path.join(config.general.setup_dir, "train"),
            val_log_step=config.train.val_log_step,
            use_auto_mixed_precision=config.train.use_auto_mixed_precision,
            use_swa=config.train.use_swa,
            swa_every_it=config.train.swa_every_it,
            swa_start_it=config.train.swa_start_it,
            swa_freq_it=config.train.swa_freq_it,
            use_grad_norm=config.train.use_grad_norm,
            save_every=config.train.checkpoint_stride) +

        # visualize
        gp.Snapshot(snapshot_datasets,
            output_dir=os.path.join(config.general.setup_dir, 'snapshots'),
            output_filename='snapshot_{iteration}.hdf',
            additional_request=snapshot_request,
            every=config.train.snapshot_stride,
            dataset_dtypes={
                maxima: np.float32
            }) +
        gp.PrintProfilingStats(every=config.train.profiling_stride)
    )

    # finalize pipeline and start training
    with gp.build(pipeline):

        logger.info("Starting training...")
        with logging_redirect_tqdm():
            for i in tqdm(range(trained_until, config.train.max_iterations)):
                start = time.time()
                pipeline.request_batch(request)
                time_of_iteration = time.time() - start

                logger.info(
                    "Batch: iteration=%d, time=%f",
                    i, time_of_iteration)


def normalize(file_source, config, raw, data_config=None):
    """Add data normalization node to pipeline

    Notes
    -----
    Which normalization method should be used?
    None/default:
        [0,1] based on data type
    minmax:
        normalize such that lower bound is at 0 and upper bound at 1
        clipping is less strict, some data might be outside of range
    percminmax:
        use precomputed percentile values for minmax normalization;
        precomputed values are stored in data_config file that has to
        be supplied; set perc_min/max to tag to be used
    mean/median
        normalize such that mean/median is at 0 and 1 std/mad is at -+1
        set perc_min/max tags for clipping beforehand
    """
    if config.train.normalization is None or \
       config.train.normalization.type == 'default':
        logger.info("default normalization")
        file_source = file_source + \
            gp.Normalize(raw,
                         factor=1.0/np.iinfo(data_config['stats']['dtype']).max
                         if data_config is not None else None)
    elif config.train.normalization.type == 'minmax':
        mn = config.train.normalization.norm_bounds[0]
        mx = config.train.normalization.norm_bounds[1]
        logger.info("minmax normalization %s %s", mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeMinMax(raw, mn=mn, mx=mx, interpolatable=False)
    elif config.train.normalization.type == 'percminmax':
        mn = data_config['stats'][config.train.normalization.perc_min]
        mx = data_config['stats'][config.train.normalization.perc_max]
        logger.info("perc minmax normalization %s %s", mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeMinMax(raw, mn=mn, mx=mx)
    elif config.train.normalization.type == 'mean':
        mean = data_config['stats']['mean']
        std = data_config['stats']['std']
        mn = data_config['stats'][config.train.normalization.perc_min]
        mx = data_config['stats'][config.train.normalization.perc_max]
        logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeMeanStd(raw, mean=mean, std=std)
    elif config.train.normalization.type == 'median':
        median = data_config['stats']['median']
        mad = data_config['stats']['mad']
        mn = data_config['stats'][config.train.normalization.perc_min]
        mx = data_config['stats'][config.train.normalization.perc_max]
        logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeMedianMad(raw, median=median, mad=mad)
    else:
        raise RuntimeError("invalid normalization method %s",
                           config.train.normalization.type)
    return file_source


def get_sources(config, raw, anchor, tracks, center_tracks, data_sources,
                val=False):
    """Create gunpowder source nodes for each data source in config

    Args
    ----
    config: TrackingConfig
        Configuration object
    raw: gp.Array
        Raw data will be stored here.
    anchor: gp.Array
        Ignore this.
    tracks: gp.Graph
        Tracks/points will be stored here
    center_tracks: gp.Graph
        Used if increased division sampling is used; division points
        will be stored here
    data_sources:
        List of data sources to use; set this to correct
        config.data.data_sources object depending on if train or val
        data should be used.
    val:
        Set to true if val data is used.
    """
    sources = []
    for ds in data_sources:
        d = ds.datafile.filename
        voxel_size = gp.Coordinate(ds.voxel_size)
        if not os.path.isdir(d):
            logger.info("trimming path %s", d)
            d = os.path.dirname(d)
        logger.info("loading data %s (val: %s)", d, val)
        # set files to use, use data_config.toml if it exists
        # otherwise check for information in config object
        if os.path.isfile(os.path.join(d, "data_config.toml")):
            data_config = load_config(os.path.join(d, "data_config.toml"))
            try:
                filename_data = os.path.join(
                    d, data_config['general']['data_file'])
            except KeyError:
                filename_data = os.path.join(
                    d, data_config['general']['zarr_file'])
            filename_tracks = os.path.join(
                d, data_config['general']['tracks_file'])
            if config.train.augment.divisions != 0.0:
                try:
                    filename_divisions = os.path.join(
                        d, data_config['general']['divisions_file'])
                except KeyError:
                    logger.warning("Cannot find divisions_file in data_config, "
                                   "falling back to using tracks_file"
                                   "(usually ok unless they are not included and "
                                   "there is a separate file containing the "
                                   "divisions)")
                    filename_divisions = os.path.join(
                        d, data_config['general']['tracks_file'])
            filename_daughters = os.path.join(
                d, data_config['general']['daughter_cells_file'])
        else:
            data_config = None
            filename_data = d
            filename_tracks = ds.tracksfile
            filename_divisions = ds.divisionsfile
            filename_daughters = ds.daughtersfile
        logger.info("creating source: %s (%s, %s, %s), divisions?: %s",
                    filename_data, ds.datafile.group,
                    filename_tracks, filename_daughters,
                    config.train.augment.divisions)
        limit_to_roi = gp.Roi(offset=ds.roi.offset, shape=ds.roi.shape)
        logger.info("limiting to roi: %s", limit_to_roi)

        datasets = {
            raw: ds.datafile.group,
            anchor: ds.datafile.group
        }
        array_specs = {
            raw: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size),
            anchor: gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size)
        }
        file_source = gp.ZarrSource(
            filename_data,
            datasets=datasets,
            nested="nested" in ds.datafile.group,
            array_specs=array_specs)

        file_source = file_source + \
            gp.Crop(raw, limit_to_roi)
        file_source = normalize(file_source, config, raw, data_config)

        file_source = file_source + \
            gp.Pad(raw, None)

        # if some frames should not be sampled from
        if ds.exclude_times:
            random_location = RandomLocationExcludeTime
            args = [raw, ds.exclude_times]
        else:
            random_location = gp.RandomLocation
            args = [raw]

        track_source = (
            merge_sources(
                file_source,
                tracks,
                center_tracks,
                filename_tracks,
                filename_tracks,
                limit_to_roi,
                use_radius=config.train.use_radius) +
            random_location(
                *args,
                ensure_nonempty=center_tracks,
                p_nonempty=config.train.augment.reject_empty_prob,
                point_balance_radius=config.train.augment.point_balance_radius)
        )

        # if division nodes should be sampled more often
        if config.train.augment.divisions != 0.0:
            div_source = (
                merge_sources(
                    file_source,
                    tracks,
                    center_tracks,
                    filename_divisions,
                    filename_daughters,
                    limit_to_roi,
                    use_radius=config.train.use_radius) +
                random_location(
                    *args,
                    ensure_nonempty=center_tracks,
                    p_nonempty=config.train.augment.reject_empty_prob,
                    point_balance_radius=config.train.augment.point_balance_radius)
            )

            track_source = (track_source, div_source) + \
                gp.RandomProvider(probabilities=[
                    1.0-config.train.augment.divisions,
                    config.train.augment.divisions])

        sources.append(track_source)

    return sources



def merge_sources(
        raw,
        tracks,
        center_tracks,
        track_file,
        center_cell_file,
        roi,
        scale=1.0,
        use_radius=False
        ):
    """Create two Track/Point sources, one with a smaller Roi.
    Goal: During sampling a random location will be selected such that
    at least one point is within the smaller Roi, but all points within
    the larger Roi will be included.
    """
    return (
        (raw,
         # tracks
         TracksSource(
                track_file,
             tracks,
             points_spec=gp.PointsSpec(roi=roi),
             scale=scale,
             use_radius=use_radius),
         # center tracks
         TracksSource(
             center_cell_file,
             center_tracks,
             points_spec=gp.PointsSpec(roi=roi),
             scale=scale,
             use_radius=use_radius),
         ) +
        gp.MergeProvider() +
        # not None padding works in combination with ensure_nonempty in
        # random_location as always a random point is picked and the roi
        # shifted such that that point is inside
        gp.Pad(tracks, gp.Coordinate((0, 500, 500, 500))) +
        gp.Pad(center_tracks, gp.Coordinate((0, 500, 500, 500)))
    )
