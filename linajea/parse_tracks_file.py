import csv

import numpy as np


def parse_csv_ndims(filename, read_dims, scale=None):
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

    locations = np.array(
        [
            [float(d) for d in line[:read_dims]]
            for line in tokens
        ], dtype=np.float32)

    track_info = np.array(
        [
            [int(i.split(".")[0]) for i in line[read_dims:ldim]]
            for line in tokens
        ], dtype=np.int32)

    if scale is not None:
        locations *= scale

    return locations, track_info

def parse_csv_fields(filename, csv_fields, scale=1.0):
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
            assert csv_fields is not None, \
                "please pass read_dims or a list of csv fieldnames to parse track file (or supply track file with headers)"
        reader = csv.DictReader(f, fieldnames=csv_fields,
                                dialect=dialect)
        for row in reader:
            loc = np.array((row['t'],
                            row['z'],
                            row['y'],
                            row['x']),
                           dtype=np.float32) * scale
            locations.append(loc)
            track_info.append([int(row['cell_id']),
                               int(row['parent_id']),
                               int(row['track_id']),
                               [row.get("radius"),
                                row.get("name")]])
            if 'div_state' in row:
                track_info[-1].append(int(row['div_state']))

    return np.array(locations), np.array(track_info)
