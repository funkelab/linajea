import csv

import numpy as np

from daisy import Coordinate

def parse_tracks_file(filename, read_dims=None, csv_fields=None, scale=1.0, limit_to_roi=None):
    if read_dims is not None:
        locations, track_info = \
            _parse_csv_ndims(filename, read_dims, scale, limit_to_roi)
    else:
        locations, track_info = \
            _parse_csv_fields(filename, csv_fields, scale, limit_to_roi)
    return locations, track_info


def _parse_csv_ndims(filename, read_dims, scale=1.0, limit_to_roi=None):
    '''Read one point per line. If ``read_dims`` is 0, all values
    in one line are considered as the location of the point. If positive,
    only the first ``ndims`` are used. If negative, all but the last
    ``-ndims`` are used.
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
            np.array([int(i.split(".")[0]) \
                      for i in line[read_dims:ldim]]))

    return np.array(locations, dtype=np.float32), \
           np.array(track_info, dtype=np.int32)

def _parse_csv_fields(filename, csv_fields, scale=1.0, limit_to_roi=None):
    '''Read one point per line. If there is no header in the file
    ``csv_fields`` is used to determine which fields are in the file
    '''
    locations = []
    track_info = []
    with open(filename, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        has_header = csv.Sniffer().has_header(f.read(1024))
        f.seek(0)
        if not has_header:
            assert csv_fields is not None and csv_fields != "header", \
                "please pass read_dims or a list of csv fieldnames to parse track file (or supply track file with headers)"
        if csv_fields == "header":
            csv_fields = None
        reader = csv.DictReader(f, fieldnames=csv_fields,
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
