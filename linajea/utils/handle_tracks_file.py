"""Provides functions to read tracks from csv text file into a dict.

The tracks dictionary (indexed by cell ID) can then be passed to a gunpowder
TracksSource node for training (parse_tracks_file_for_tracks_source) or stored
as ground truth in a database (add_tracks_to_database)


Note
----
The CSV file must have a header!
"""
import csv
import logging

import numpy as np

from daisy import Roi, Coordinate

from linajea.utils import CandidateDatabase

logger = logging.getLogger(__name__)


def add_tracks_to_database(
        csv_tracks_file,
        db_name,
        db_host,
        limit_to_roi=None):
    """Loads tracks file and add tracks to a mongodb database

    Args:
    -----
    csv_tracks_file: str
        CSV containing the tracks. See `_load_csv_to_dict` for a description
        of the required content.
    db_name: str
        Store tracks into this databse.
    db_host: str
        Address of mongodb server.
    limit_to_roi: ROI, optional
        Optional Region of Interest, only return cells within region.
    """
    cells = _load_csv_to_dict(csv_tracks_file)

    db = CandidateDatabase(db_name, db_host, mode='w')
    nodes = {}
    edges = []
    min_dims = None
    max_dims = None
    for cell_tmp in cells.values():
        cell = {}
        for key, val in cell_tmp.items():
            if key in ('z', 'y', 'x', 'radius'):
                val = float(val)
            elif key in ('div_state',
                         'cell_id', 'parent_id', 'track_id'):
                val = int(val)
            elif key == 't':
                val = int(float(val))
            if key == 'cell_id':
                key = 'id'
            elif key == 'radius':
                key = 'r'
            cell[key] = val

        position = [cell['t'], cell['z'], cell['y'], cell['x']]
        if limit_to_roi is not None and \
           not limit_to_roi.contains(Coordinate(position)):
            continue

        if min_dims is None:
            min_dims = position
        else:
            min_dims = [min(prev_min, curr)
                        for prev_min, curr in zip(min_dims, position)]

        if max_dims is None:
            max_dims = position
        else:
            max_dims = [max(prev_max, curr)
                        for prev_max, curr in zip(max_dims, position)]

        nodes[cell['id']] = cell

    for cell in nodes.values():
        if cell['parent_id'] in nodes:
            edges.append((cell['id'], cell['parent_id']))

    assert nodes, f"Did not find any nodes in file {csv_tracks_file}"

    min_dims = Coordinate(min_dims)
    max_dims = Coordinate(max_dims)
    roi = Roi(min_dims, max_dims - min_dims + Coordinate((1, 1, 1, 1)))
    subgraph = db[roi]
    subgraph.add_nodes_from(zip(nodes.keys(), nodes.values()))
    subgraph.add_edges_from(edges)
    subgraph.write_nodes()
    subgraph.write_edges()
    logger.info("Added %s nodes and %s edges",
                len(subgraph.nodes), len(subgraph.edges))


def parse_tracks_file_for_tracks_source(
        csv_tracks_file,
        scale=1.0,
        limit_to_roi=None,
        attr_filter={}):
    """Loads tracks file and converts tracks into format expected by
    the gunpowder TracksSource node

    Args:
    -----
    csv_tracks_file: str
        CSV containing the tracks. See `_load_csv_to_dict` for a description
        of the required content.
    limit_to_roi: ROI, optional
        Optional Region of Interest, only return cells within region.
    attr_filter: dict
        Discard cells that do not have attr=value set for each element in
        attr_filter.
    """
    cells = _load_csv_to_dict(csv_tracks_file)
    _set_division_state(cells)

    locations = []
    track_info = []
    for cell in cells.values():
        loc = np.array((cell['t'],
                        cell['z'],
                        cell['y'],
                        cell['x']),
                       dtype=np.float32) * scale
        if limit_to_roi is not None and \
           not limit_to_roi.contains(Coordinate(loc)):
            continue
        if attr_filter:
            discard = False
            for k, v in attr_filter.items():
                if cell[k] != v:
                    discard = True
                    break
            if discard:
                continue

        locations.append(loc)

        ti = {"cell_id": int(cell['cell_id']),
              "parent_id": int(cell['parent_id']),
              "track_id": int(cell['track_id'])}
        attrs = {}
        if "radius" in cell:
            attrs["radius"] = float(cell['radius'])
        if "name" in cell:
            attrs["name"] = cell['name']
        if 'div_state' in cell:
            attrs["div_state"] = int(cell['div_state'])
        ti["attrs"] = attrs
        track_info.append(ti)

    return np.array(locations), np.array(track_info, dtype=object)


def _load_csv_to_dict(csv_tracks_file):
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
    where ``parent_id`` can be -1 to indicate no parent.

    Args:
    -----
    csv_tracks_file: str
        The file to read the tracks from

    Returns:
    --------
    cells: dict int: cells (dict str: value)
        Dictionary of the cells contained in the csv file with the cell_id as
        key.
    '''
    dialect, has_header = _get_dialect_and_header(csv_tracks_file)
    assert has_header, "Please provide a tracks file with a header line"

    cells = {}
    with open(csv_tracks_file, 'r', newline='') as fl:
        reader = csv.DictReader(fl, dialect=dialect)

        for cell in reader:
            cells[cell['cell_id']] = cell

    return cells


def _set_division_state(cells):
    """Internal function to compute the division state for each cell based on
    the contained children/sibling information
    """
    parent_child_dict = {}

    for cell_id, cell in cells.items():
        parent_id = int(cell['parent_id'])
        if parent_id == -1:
            continue
        if parent_id not in parent_child_dict:
            parent_child_dict[parent_id] = []
        parent_child_dict[parent_id].append(cell)

    for cell_id, cell in cells.items():
        if 'div_state' in cell:
            continue

        parent_id = int(cell['parent_id'])
        if parent_id == -1:
            siblings = [cell]
        else:
            siblings = parent_child_dict[parent_id]
        children = parent_child_dict.get(cell_id, [])

        assert len(children) <= 2, \
            "Error! Cell has more than two children: %d" % cell_id
        assert len(siblings) <= 2, \
            "Error! Cell has more than two siblings: %d" % parent_id
        assert len(siblings) > 0, \
            "Error! Cell has no children but should have one: %d" % parent_id
        assert not (
            len(siblings) == 2 and
            len(children) == 2), (
                "Error! No timepoint between to successive divisions: "
                f"{parent_id}, {cell_id}")

        if len(children) == 2:
            cell['div_state'] = 1
        elif parent_id == -1:
            cell['div_state'] = 0
        elif len(siblings) == 2:
            cell['div_state'] = 2
        else:  # if len(siblings) == 1 and  len(children) == 1:
            cell['div_state'] = 0


def _get_dialect_and_header(csv_file):
    """Get the CSV dialect (e.g. delimiter) and check if the file contains a
    header line"""
    with open(csv_file, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        has_header = csv.Sniffer().has_header(f.read(1024))

    return dialect, has_header
