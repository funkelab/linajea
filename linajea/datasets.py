import daisy
import json
import os

def get_source_roi(data_dir, sample):

    sample_path = os.path.join(data_dir, sample)

    # get absolute paths
    if os.path.isfile(sample_path) or sample.endswith((".zarr", ".n5")):
        sample_dir = os.path.abspath(os.path.join(data_dir,
                                                  os.path.dirname(sample)))
    else:
        sample_dir = os.path.abspath(os.path.join(data_dir, sample))

    if os.path.isfile(os.path.join(sample_dir, 'attributes.json')):

        with open(os.path.join(sample_dir, 'attributes.json'), 'r') as f:
            attributes = json.load(f)
        voxel_size = daisy.Coordinate(attributes['resolution'])
        shape = daisy.Coordinate(attributes['shape'])
        offset = daisy.Coordinate(attributes['offset'])
        source_roi = daisy.Roi(offset, shape*voxel_size)

        return voxel_size, source_roi

    elif os.path.isdir(os.path.join(sample_dir, 'timelapse.zarr')):

        a = daisy.open_ds(
            os.path.join(sample_dir, 'timelapse.zarr'),
            'volumes/raw')

        return a.voxel_size, a.roi

    else:

        raise RuntimeError(
            "Can't find attributes.json or timelapse.zarr in %s" %
            sample_dir)

