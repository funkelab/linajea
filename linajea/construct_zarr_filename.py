import os


def construct_zarr_filename(config, sample, checkpoint):
    return os.path.join(
        config.predict.output_zarr_prefix,
        config.general.setup,
        os.path.basename(sample) +
        'predictions' + (config.general.tag if config.general.tag is not None else "") +
        str(checkpoint) + "_" +
        str(config.inference.cell_score_threshold).replace(".", "_") + '.zarr')
