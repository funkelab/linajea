import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import logging
import os


import h5py
import numpy as np

import gunpowder as gp

from linajea.config import TrackingConfig
from linajea.gunpowder_nodes import WriteCells
from linajea.process_blockwise import write_done
from linajea.utils import (load_config,
                           construct_zarr_filename)


logger = logging.getLogger(__name__)


def write_cells_from_zarr(config):

    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    if not config.model.train_only_cell_indicator:
        parent_vectors = gp.ArrayKey('PARENT_VECTORS')

    voxel_size = gp.Coordinate(config.inference_data.data_source.voxel_size)
    output_size = gp.Coordinate(output_shape) * voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)
    if not config.model.train_only_cell_indicator:
        chunk_request.add(parent_vectors, output_size)

    sample = config.inference_data.data_source.datafile.filename
    if os.path.isfile(os.path.join(sample, "data_config.toml")):
        data_config = load_config(
            os.path.join(sample, "data_config.toml"))
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

    output_path = construct_zarr_filename(
        config,
        config.inference_data.data_source.datafile.filename,
        config.inference_data.checkpoint)

    datasets = {
        cell_indicator: 'volumes/cell_indicator',
        maxima: '/volumes/maxima'}
    if not config.model.train_only_cell_indicator:
        datasets[parent_vectors] = 'volumes/parent_vectors'

    array_specs = {
        cell_indicator: gp.ArraySpec(
            interpolatable=True,
            voxel_size=voxel_size),
        maxima: gp.ArraySpec(
            interpolatable=False,
            voxel_size=voxel_size)}
    if not config.model.train_only_cell_indicator:
        array_specs[parent_vectors] = gp.ArraySpec(
            interpolatable=True,
            voxel_size=voxel_size)

    source = gp.ZarrSource(
        output_path,
        datasets=datasets,
        array_specs=array_specs)

    roi_map = {
        cell_indicator: 'write_roi',
        maxima: 'write_roi'
    }
    if not config.model.train_only_cell_indicator:
        roi_map[parent_vectors] = 'write_roi'

    pipeline = (
        source +
        gp.Pad(cell_indicator, size=None) +
        gp.Pad(maxima, size=None))

    if not config.model.train_only_cell_indicator:
        pipeline = (pipeline +
                    gp.Pad(parent_vectors, size=None))

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
            volume_shape=volume_shape) +
        gp.PrintProfilingStats(every=100) +
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map=roi_map,
            num_workers=1,
            block_done_callback=lambda b, st, et: write_done(
                b,
                'predict_db',
                config.inference_data.data_source.db_name,
                config.general.db_host)
    ))

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')

    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)
    write_cells_sample(config)
