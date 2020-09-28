import os

def construct_zarr_filename(config, sample):
    return os.path.join(
        config['general']['output_zarr_prefix'],
        os.path.basename(config['general']['setup_dir']),
        os.path.basename(sample) +
        'predictions' +
        str(config['prediction']['iteration']) + '.zarr')
