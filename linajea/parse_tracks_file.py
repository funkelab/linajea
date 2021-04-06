import csv
import logging

import numpy as np

from daisy import Coordinate

logger = logging.getLogger(__name__)


def get_dialect_and_header(csv_file):
    with open(csv_file, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        has_header = csv.Sniffer().has_header(f.read(1024))

    return dialect, has_header


def parse_tracks_file(
        filename,
        scale=1.0,
        limit_to_roi=None):
    dialect, has_header = get_dialect_and_header(filename)
    logger.debug("Tracks file has header: %s" % has_header)
    if has_header:
        locations, track_info = \
            _parse_csv_fields(filename, scale, limit_to_roi)
    else:
        locations, track_info = \
            _parse_csv_ndims(filename, scale, limit_to_roi)
    return locations, track_info


def _parse_csv_ndims(filename, scale=1.0, limit_to_roi=None, read_dims=4):
    '''Read one point per line. If ``read_dims`` is 0, all values
    in one line are considered as the location of the point. If positive,
    only the first ``ndims`` are used. If negative, all but the last
    ``-ndims`` are used. Defaults to 4, so the first 4 values are considered
    locations t z y x.
    '''
    with open(filename, 'r') as f:
        tokens = [[t.strip(',') for t in line.split()]
                  for line in f]
    try:
        _ = int(tokens[0][-1])
        ldim = None
    except ValueError:
        ldim = -1

    locations = []
    track_info = []
    for line in tokens:
        loc = np.array([float(d) for d in line[:read_dims]]) * scale
        if limit_to_roi is not None and \
           not limit_to_roi.contains(Coordinate(loc)):
            continue

        locations.append(loc)
        track_info.append(
            np.array([int(i.split(".")[0])
                      for i in line[read_dims:ldim]]))

    return (np.array(locations, dtype=np.float32),
            np.array(track_info, dtype=np.int32))


def _parse_csv_fields(filename, scale=1.0, limit_to_roi=None):
    '''Read one point per line. Assumes a header with the following required
    fields:
        t
        z
        y
        x
        cell_id
        parent_id
        track_id
    And these optional fields:
        radius
        name
        div_state
    '''
    locations = []
    track_info = []
    dialect, has_header = get_dialect_and_header(filename)
    with open(filename, 'r') as f:
        assert has_header, "No header found, but this function needs a header"
        reader = csv.DictReader(f, fieldnames=None,
                                dialect=dialect)
        for row in reader:
            loc = np.array((row['t'],
                            row['z'],
                            row['y'],
                            row['x']),
                           dtype=np.float32) * scale
            if limit_to_roi is not None and \
               not limit_to_roi.contains(Coordinate(loc)):
                continue
            locations.append(loc)
            track_info.append([int(row['cell_id']),
                               int(row['parent_id']),
                               int(row['track_id']),
                               [row.get("radius"),
                                row.get("name")]])
            if 'div_state' in row:
                track_info[-1].append(int(row['div_state']))

    return np.array(locations), np.array(track_info)
