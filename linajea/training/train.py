"""Script for training process

Create model and train
"""
import logging
import time
import os

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import gunpowder as gp

from linajea.gunpowder_nodes import (
    AddMovementVectors,
    ElasticAugment,
    HistogramAugment,
    NoiseAugment,
    NoOp,
    RandomLocationExcludeTime,
    RasterizeGraph,
    ShiftAugment,
    ShuffleChannels,
    TracksSource,
    TorchTrainExt,
    TrainValProvider,
    ZoomAugment)
from linajea.config import load_config

from . import torch_model
from . import torch_loss
from .utils import (
    get_latest_checkpoint,
    normalize)

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
    checkpoint_basename = 'train_net'
    _, trained_until = get_latest_checkpoint(checkpoint_basename)
    # training already done?
    if trained_until >= config.train.max_iterations:
        logger.info(
            "Model has already been trained for %s iterations", trained_until)
        return

    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    tracks = gp.GraphKey('TRACKS')
    center_tracks = gp.GraphKey('CENTER_TRACKS')
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

    input_shape, output_shape_1, output_shape_2 = model.inout_shapes(
        device=device)
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

    train_sources = get_sources(config, raw, tracks, center_tracks,
                                config.train_data.data_sources, val=False)

    # Do interleaved validation?
    if config.train.val_log_step is not None:
        val_sources = get_sources(config, raw, tracks, center_tracks,
                                  config.validate_data.data_sources, val=True)

    # set up pipeline:
    # load data source and
    # choose augmentations depending on config
    augment = config.train.augment
    use_fast_points_transform = (
        augment.elastic.use_fast_points_transform
        if augment.elastic is not None else None)

    train_pipeline = (
        tuple(train_sources) +
        gp.RandomProvider() +

        (ElasticAugment(
            augment.elastic.control_point_spacing,
            augment.elastic.jitter_sigma,
            [augment.elastic.rotation_min*np.pi/180.0,
             augment.elastic.rotation_max*np.pi/180.0],
            rotation_3d=augment.elastic.rotation_3d,
            subsample=augment.elastic.subsample,
            use_fast_points_transform=use_fast_points_transform,
            spatial_dims=3,
            temporal_dim=True)
         if augment.elastic is not None else NoOp()) +

        (ShiftAugment(
            prob_slip=augment.shift.prob_slip,
            prob_shift=augment.shift.prob_shift,
            sigma=augment.shift.sigma,
            shift_axis=0)
         if augment.shift is not None else NoOp()) +

        (ShuffleChannels(raw)
         if augment.shuffle_channels else NoOp()) +

        (gp.SimpleAugment(
            mirror_only=augment.simple.mirror,
            transpose_only=augment.simple.transpose)
         if augment.simple is not None else NoOp()) +

        (ZoomAugment(
            factor_min=augment.zoom.factor_min,
            factor_max=augment.zoom.factor_max,
            spatial_dims=augment.zoom.spatial_dims,
            order={raw: 1,
                   })
         if augment.zoom is not None else NoOp()) +

        (NoiseAugment(
            raw,
            mode='gaussian',
            var=augment.noise_gaussian.var,
            clip=False,
            check_val_range=False)
         if augment.noise_gaussian is not None else NoOp()) +

        (NoiseAugment(
            raw,
            mode='speckle',
            var=augment.noise_speckle.var,
            clip=False,
            check_val_range=False)
         if augment.noise_speckle is not None else NoOp()) +

        (NoiseAugment(
            raw,
            mode='s&p',
            amount=augment.noise_saltpepper.amount,
            clip=False,
            check_val_range=False)
         if augment.noise_saltpepper is not None else NoOp()) +

        (HistogramAugment(
            raw,
            # raw_tmp,
            range_low=augment.histogram.range_low,
            range_high=augment.histogram.range_high,
            z_section_wise=False)
         if augment.histogram is not None else NoOp()) +

        (gp.IntensityAugment(
            raw,
            scale_min=augment.intensity.scale[0],
            scale_max=augment.intensity.scale[1],
            shift_min=augment.intensity.shift[0],
            shift_max=augment.intensity.shift[1],
            z_section_wise=False,
            clip=False)
         if augment.intensity is not None else NoOp()) +

        (AddMovementVectors(
            tracks,
            movement_vectors,
            cell_mask,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            object_radius=config.train.object_radius,
            move_radius=config.train.move_radius,
            dense=not config.general.sparse)
         if not config.model.train_only_cell_indicator else NoOp()) +

        (gp.Reject(
            ensure_nonempty=tracks,
            mask=cell_mask,
            min_masked=0.0001)
         if config.general.sparse else NoOp()) +

        RasterizeGraph(
            tracks,
            cell_indicator,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=config.train.rasterize_radius,
                mode='peak')) +

        RasterizeGraph(
            tracks,
            cell_center,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.1,) + voxel_size[1:],
                mode='point'))
        +

        (gp.PreCache(
            cache_size=config.train.cache_size,
            num_workers=config.train.job.num_workers)
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
                object_radius=config.train.object_radius,
                move_radius=config.train.move_radius,
                dense=not config.general.sparse)
             if not config.model.train_only_cell_indicator else NoOp()) +

            (gp.Reject(
                ensure_nonempty=tracks,
                mask=cell_mask,
                min_masked=0.0001,
                reject_probability=augment.reject_empty_prob)
             if config.general.sparse else NoOp()) +

            gp.Reject(
                ensure_nonempty=center_tracks,
                # always reject emtpy batches in validation branch
                reject_probability=1.0
            ) +

            RasterizeGraph(
                tracks,
                cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=config.train.rasterize_radius,
                    mode='peak')) +

            RasterizeGraph(
                tracks,
                cell_center,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=(0.1,) + voxel_size[1:],
                    mode='point'))
            +

            (gp.PreCache(
                cache_size=config.train.cache_size,
                num_workers=1)
             if config.train.job.num_workers > 1 else NoOp())
        )

    if config.train.val_log_step is not None:
        pipeline = (
            (train_pipeline, val_pipeline) +
            TrainValProvider(
                step=config.train.val_log_step, init_step=trained_until))
    else:
        pipeline = train_pipeline

    inputs = {
        'raw': raw,
    }
    if not config.model.train_only_cell_indicator:
        inputs['cell_mask'] = cell_mask
        inputs['gt_movement_vectors'] = movement_vectors

    outputs = {
        0: pred_cell_indicator,
        1: maxima,
        2: raw_cropped,
    }
    if not config.model.train_only_cell_indicator:
        outputs[3] = pred_movement_vectors

    loss_inputs = {
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
        snapshot_datasets[pred_movement_vectors] = \
            'volumes/pred_movement_vectors'
        snapshot_datasets[grad_movement_vectors] = \
            'volumes/grad_movement_vectors'

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
            "train_net_checkpoint_0")

    # create loss object
    loss = torch_loss.LossWrapper(config, current_step=trained_until)

    # and add training gunpowder node
    pipeline = (
        pipeline +
        TorchTrainExt(
            model=model,
            loss=loss,
            optimizer=opt,
            checkpoint_basename=checkpoint_basename,
            inputs=inputs,
            outputs=outputs,
            loss_inputs=loss_inputs,
            gradients=gradients,
            log_dir="train",
            val_log_step=config.train.val_log_step,
            use_auto_mixed_precision=config.train.use_auto_mixed_precision,
            use_swa=config.train.use_swa,
            swa_every_it=config.train.swa_every_it,
            swa_start_it=config.train.swa_start_it,
            swa_freq_it=config.train.swa_freq_it,
            save_every=config.train.checkpoint_stride) +

        # visualize
        gp.Snapshot(snapshot_datasets,
                    output_dir='snapshots',
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


def get_sources(config, raw, tracks, center_tracks, data_sources,
                val=False):
    """Create gunpowder source nodes for each data source in config

    Args
    ----
    config: TrackingConfig
        Configuration object
    raw: gp.Array
        Raw data will be stored here.
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
        filename_data = ds.datafile.filename
        filename_tracks = ds.tracksfile
        file_attrs = daisy.open_ds(
            filename_data, ds.datafile.array, 'r').data.attrs
        voxel_size = gp.Coordinate(ds.voxel_size)
        logger.info("loading data %s (val: %s)", filename_data, val)
        logger.info("creating source: %s (%s, %s), divisions?: %s",
                    filename_data, ds.datafile.array,
                    filename_tracks,
                    config.train.augment.divisions)
        limit_to_roi = gp.Roi(offset=ds.roi.offset, shape=ds.roi.shape)
        logger.info("limiting to roi: %s", limit_to_roi)

        datasets = {
            raw: ds.datafile.array
        }
        array_specs = {
            raw: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size)
        }
        file_source = gp.ZarrSource(
            filename_data,
            datasets=datasets,
            array_specs=array_specs)

        file_source = file_source + \
            gp.Crop(raw, limit_to_roi)
        file_source = normalize(
            file_source, config.train.normalization, raw, file_attrs)

        file_source = file_source + \
            gp.Pad(raw, None)

        # if some frames should not be sampled from
        if ds.exclude_times:
            random_location = RandomLocationExcludeTime
            args = [raw, ds.exclude_times]
        else:
            random_location = gp.RandomLocation
            args = [raw]

        point_balance_radius = config.train.augment.point_balance_radius
        track_source = (
            merge_sources(
                file_source,
                tracks,
                center_tracks,
                filename_tracks,
                limit_to_roi,
                use_radius=config.train.use_radius) +
            random_location(
                *args,
                ensure_nonempty=center_tracks,
                p_nonempty=config.train.augment.reject_empty_prob,
                point_balance_radius=point_balance_radius)
        )

        # if division nodes should be sampled more often
        if config.train.augment.divisions != 0.0:
            file_sourceD = gp.ZarrSource(
                filename_data,
                datasets=datasets,
                array_specs=array_specs)

            file_sourceD = file_sourceD + \
                gp.Crop(raw, limit_to_roi)
            file_sourceD = normalize(
                file_sourceD, config.train.normalization, raw, file_attrs)

            file_sourceD = file_sourceD + \
                gp.Pad(raw, None)

            div_source = (
                merge_sources(
                    file_sourceD,
                    tracks,
                    center_tracks,
                    filename_tracks,
                    limit_to_roi,
                    use_radius=config.train.use_radius,
                    attr_filter={"div_state": 2}) +
                random_location(
                    *args,
                    ensure_nonempty=center_tracks,
                    p_nonempty=config.train.augment.reject_empty_prob,
                    point_balance_radius=point_balance_radius)
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
        csv_tracks_file,
        roi,
        scale=1.0,
        use_radius=False,
        attr_filter={}
        ):
    """Create two Track/Point sources, one with a smaller Roi.
    Goal: During sampling a random location will be selected such that
    at least one point is within the smaller Roi, but all points within
    the larger Roi will be included.

    Args:
    -----
    raw: gp.ArrayKey
        Represents a gp.Array that will contain the raw image data
    tracks: gp.GraphKey
        Represents a gp.Graph that will contain the points of all cells in
        the input ROI
    center_tracks: gp.GraphKey
        Represents a gp.Graph with a slightly smaller ROI than `tracks`.
        In each iteration select a random location such that at least one
        cell lies in this ROI (to avoid having only a single cell right on
        the border of the larger ROI that might get cropped in later stages)
    csv_tracks_file: Path
        Load the tracks from this file. Check
        `utils/handle_tracks_file.py:_load_csv_to_dict` for more information
        on the required format of the file.
    roi: ROI
        Restrict data to this ROI
    scale: scalar or array-like
        An optional scaling to apply to the coordinates of the points read
        from the CSV file. This is useful if the points refer to voxel
        positions to convert them to world units.
    use_radius: Boolean or dict int: int
        If True, use the radii information contained in the CSV file.
        If a dict, the keys refer to (sparse) frame indices and the values to
        radii; assign to each cell the radius associated with the next larger
        frame index, e.g. `{30: 15, 50: 7}`, all cells before frame 30 get
        radius 15 and all cells after 30 but before 50 get radius 7.
    attr_filter: dict
        Only consider cells for the `center_tracks` Graph that have
        attr=value set for each element in attr_filter.
    """
    return (
        (raw,
         # tracks
         TracksSource(
             csv_tracks_file,
             tracks,
             points_spec=gp.GraphSpec(roi=roi),
             scale=scale,
             use_radius=use_radius),
         # center tracks
         TracksSource(
             csv_tracks_file,
             center_tracks,
             points_spec=gp.GraphSpec(roi=roi),
             scale=scale,
             use_radius=use_radius,
             attr_filter=attr_filter),
         ) +
        gp.MergeProvider() +
        # not None padding works in combination with ensure_nonempty in
        # random_location as always a random point is picked and the roi
        # shifted such that that point is inside
        gp.Pad(tracks, gp.Coordinate((0, 500, 500, 500))) +
        gp.Pad(center_tracks, gp.Coordinate((0, 500, 500, 500)))
    )
