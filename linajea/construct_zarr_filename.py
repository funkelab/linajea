import os


def construct_zarr_filename(config, sample, checkpoint):
    return os.path.join(
        config.predict.output_zarr_prefix,
        config.general.setup,
        os.path.basename(sample) +
        'predictions' +
        str(checkpoint) + '.zarr')
