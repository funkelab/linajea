"""Provides function to read tracks from csv text file

First checks if file has header, if yes uses it to parse file,
if no assumes default order of columns
"""
import csv
import logging

import numpy as np

from daisy import Coordinate

logger = logging.getLogger(__name__)


def parse_tracks_file(
        filename,
        scale=1.0,
        limit_to_roi=None):
    '''Read one point per line. Expects a header with the following required
    fields:
        t
        z
        y
        x
        cell_id
        parent_id
        track_id
    And optional fields such as:
        radius
        name
        div_state
    '''
    dialect, has_header = _get_dialect_and_header(filename)
    assert has_header, "Please provide a tracks file with a header line"

    locations = []
    track_info = []
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
            ti = {"cell_id": int(row['cell_id']),
                  "parent_id": int(row['parent_id']),
                  "track_id": int(row['track_id'])}
            attrs = {}
            if "radius" in row:
                attrs["radius"] = float(row['radius'])
            if "name" in row:
                attrs["name"] = row['name']
            if 'div_state' in row:
                attrs["div_state"] = int(row['div_state'])
            ti["attrs"] = attrs
            track_info.append(ti)

    return np.array(locations), np.array(track_info, dtype=object)


def _get_dialect_and_header(csv_file):
    with open(csv_file, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        has_header = csv.Sniffer().has_header(f.read(1024))

    return dialect, has_header
