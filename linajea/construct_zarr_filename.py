import os

def construct_zarr_filename(**kwargs):
    return os.path.join(
        kwargs['general']['output_zarr_prefix'],
        os.path.basename(kwargs['general']['setup_dir']),
        os.path.basename(kwargs['sample']) +
        'predictions' +
        str(kwargs['prediction']['iteration']) + '.zarr')
