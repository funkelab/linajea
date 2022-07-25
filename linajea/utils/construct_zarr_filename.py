"""Provide simple function to create standard filename for zarr prediction file
"""
import os


def construct_zarr_filename(config, sample, checkpoint):
    return os.path.join(
        config.predict.output_zarr_dir,
        os.path.basename(config.general.setup_dir),
        os.path.basename(sample) +
        'predictions' + (config.general.tag if config.general.tag is not None
                         else "") +
        str(checkpoint) + "_" +
        str(config.inference_data.cell_score_threshold).replace(".", "_") +
        '.zarr')
