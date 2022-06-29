import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import logging
import os

import h5py
import numpy as np
import torch

import daisy
import gunpowder as gp

from linajea.config import (load_config,
                            TrackingConfig)
from linajea.gunpowder_nodes import (Clip,
                                     NormalizeMinMax,
                                     NormalizeMeanStd,
                                     NormalizeMedianMad,
                                     WriteCells)
from linajea.process_blockwise import write_done
import linajea.training.torch_model
from linajea.utils import construct_zarr_filename


logger = logging.getLogger(__name__)


def predict(config):

    raw = gp.ArrayKey('RAW')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    if not config.model.train_only_cell_indicator:
        parent_vectors = gp.ArrayKey('PARENT_VECTORS')

    model = linajea.training.torch_model.UnetModelWrapper(
        config, config.inference_data.checkpoint)
    model.eval()
    logger.info("Model: %s", model)

    input_shape = config.model.predict_input_shape
    trial_run = model.forward(torch.zeros(input_shape, dtype=torch.float32))
    _, _, trial_max, _ = trial_run
    output_shape = trial_max.size()

    voxel_size = gp.Coordinate(config.inference_data.data_source.voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)
    if not config.model.train_only_cell_indicator:
        chunk_request.add(parent_vectors, output_size)

    sample = config.inference_data.data_source.datafile.filename
    if os.path.isfile(os.path.join(sample, "data_config.toml")):
        data_config = load_config(
            os.path.join(sample, "data_config.toml"))
        try:
            filename_data = os.path.join(
                sample, data_config['general']['data_file'])
        except KeyError:
            filename_data = os.path.join(
                sample, data_config['general']['zarr_file'])
        filename_mask = os.path.join(
            sample,
            data_config['general'].get('mask_file', os.path.splitext(
                    data_config['general']['zarr_file'])[0] + "_mask.hdf"))
        z_range = data_config['general']['z_range']
        if z_range[1] < 0:
            z_range[1] = data_config['general']['shape'][1] - z_range[1]
        volume_shape = data_config['general']['shape']
    else:
        data_config = None
        filename_data = sample
        filename_mask = sample + "_mask.hdf"
        z_range = None
        volume_shape = daisy.open_ds(
            filename_data,
            config.inference_data.data_source.datafile.group).roi.get_shape()

    if os.path.isfile(filename_mask):
        with h5py.File(filename_mask, 'r') as f:
            mask = np.array(f['volumes/mask'])
    else:
        mask = None

    source = gp.ZarrSource(
        filename_data,
        datasets={
            raw: config.inference_data.data_source.datafile.group
        },
        nested="nested" in config.inference_data.data_source.datafile.group,
        array_specs={
            raw: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size)})

    source = normalize(source, config, raw, data_config)

    inputs={
        'raw': raw
    }
    outputs={
        0: cell_indicator,
        1: maxima,
    }
    if not config.model.train_only_cell_indicator:
        outputs[3] = parent_vectors

    dataset_names={
        cell_indicator: 'volumes/cell_indicator',
    }
    if not config.model.train_only_cell_indicator:
        dataset_names[parent_vectors] = 'volumes/parent_vectors'

    pipeline = (
        source +
        gp.Pad(raw, size=None) +
        gp.torch.Predict(
            model=model,
            checkpoint=os.path.join(config.general.setup_dir,
                                    'train_net_checkpoint_{}'.format(
                                        config.inference_data.checkpoint)),
            inputs=inputs,
            outputs=outputs,
            use_swa=config.predict.use_swa
        ))

    cb = []
    if config.predict.write_to_zarr:
        pipeline = (
            pipeline +

            gp.ZarrWrite(
                dataset_names=dataset_names,
                output_filename=construct_zarr_filename(config, sample,
                                                        config.inference_data.checkpoint)
            ))
        if not config.predict.no_db_access:
            cb.append(lambda b: write_done(
                b,
                'predict_zarr',
                config.inference_data.data_source.db_name,
                config.general.db_host))
        else:
            cb.append(lambda _: True)

    if config.predict.write_to_db:
        pipeline = (
            pipeline +

            WriteCells(
                maxima,
                cell_indicator,
                parent_vectors if not config.model.train_only_cell_indicator else None,
                score_threshold=config.inference_data.cell_score_threshold,
                db_host=config.general.db_host,
                db_name=config.inference_data.data_source.db_name,
                mask=mask,
                z_range=z_range,
                volume_shape=volume_shape)
            )
        cb.append(lambda b: write_done(
            b,
            'predict_db',
            db_name=config.inference_data.data_source.db_name,
            db_host=config.general.db_host))


    roi_map = {
        raw: 'read_roi',
        cell_indicator: 'write_roi',
        maxima: 'write_roi'
    }
    if not config.model.train_only_cell_indicator:
        roi_map[parent_vectors] = 'write_roi'

    pipeline = (
        pipeline +

        gp.PrintProfilingStats(every=100) +
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map=roi_map,
            num_workers=1,
            block_done_callback=lambda b, st, et: all([f(b) for f in cb])
    ))

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


def normalize(file_source, config, raw, data_config=None):
    if config.predict.normalization is None or \
       config.predict.normalization.type == 'default':
        logger.info("default normalization")
        file_source = file_source + \
            gp.Normalize(raw,
                         factor=1.0/np.iinfo(data_config['stats']['dtype']).max)
    elif config.predict.normalization.type == 'minmax':
        mn = config.predict.normalization.norm_bounds[0]
        mx = config.predict.normalization.norm_bounds[1]
        logger.info("minmax normalization %s %s", mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeMinMax(raw, mn=mn, mx=mx, interpolatable=False)
    elif config.predict.normalization.type == 'percminmax':
        mn = data_config['stats'][config.predict.normalization.perc_min]
        mx = data_config['stats'][config.predict.normalization.perc_max]
        logger.info("perc minmax normalization %s %s", mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeMinMax(raw, mn=mn, mx=mx)
    elif config.predict.normalization.type == 'mean':
        mean = data_config['stats']['mean']
        std = data_config['stats']['std']
        mn = data_config['stats'][config.predict.normalization.perc_min]
        mx = data_config['stats'][config.predict.normalization.perc_max]
        logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeMeanStd(raw, mean=mean, std=std)
    elif config.predict.normalization.type == 'median':
        median = data_config['stats']['median']
        mad = data_config['stats']['mad']
        mn = data_config['stats'][config.predict.normalization.perc_min]
        mx = data_config['stats'][config.predict.normalization.perc_max]
        logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
        file_source = file_source + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeMedianMad(raw, median=median, mad=mad)
    else:
        raise RuntimeError("invalid normalization method %s",
                           config.predict.normalization.type)
    return file_source


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)
    predict(config)
