"""Provides a gunpowder source node for tracks
"""
import logging

import numpy as np

from gunpowder import (Node, Coordinate, Batch, BatchProvider,
                       Roi, GraphSpec, Graph)
from gunpowder.profiling import Timing

from linajea.utils import parse_tracks_file_for_tracks_source

logger = logging.getLogger(__name__)


class TrackNode(Node):
    """Specializes gp.Node to set a number of attributes

    Attributes
    ----------
    original_location: np.ndarray
        location of node
    parent_id: int
        id of parent node in track that node is part of
    track_id: int
        id of track that node is part of
    value: object
        some value that should be associated with node,
        e.g., used to store object specific radius
    """
    def __init__(self, id, location, parent_id, track_id, value=None):

        attrs = {"original_location": np.array(location, dtype=np.float32),
                 "parent_id": parent_id,
                 "track_id": track_id,
                 "value": value}
        super(TrackNode, self).__init__(id, location, attrs=attrs)


class TracksSource(BatchProvider):
    '''Gunpowder source node: read tracks of points from a csv file.

    Notes:

        Check `utils/handle_tracks_file.py:_load_csv_to_dict` for more
        information on the required format of the file.

    Args:

        filename (``string``):

            The file to read from.

        points (:class:`GraphKey`):

            The key of the points set to create.

        points_spec (:class:`GraphSpec`, optional):

            An optional :class:`GraphSpec` to overwrite the points specs
            automatically determined from the CSV file. This is useful to set
            the :class:`Roi` manually, for example.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the points read
            from the CSV file. This is useful if the points refer to voxel
            positions to convert them to world units.

        use_radius (Boolean or dict int: int)

            If True, use the radii information contained in the CSV file.
            If a dict, the keys refer to (sparse) frame indices and the values
            to radii; assign to each cell the radius associated with the next
            larger frame index, e.g. `{30: 15, 50: 7}`, all cells before frame
            30 get radius 15 and all cells after 30 but before 50 get radius 7.

        attr_filter (dict)
            Only consider cells for the `center_tracks` Graph that have
            attr=value set for each element in attr_filter.
    '''

    def __init__(self,
                 filename,
                 points,
                 points_spec=None,
                 scale=1.0,
                 use_radius=False,
                 attr_filter={}):

        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        if isinstance(use_radius, dict):
            self.use_radius = {int(k): v for k, v in use_radius.items()}
        else:
            self.use_radius = use_radius
        self.attr_filter = attr_filter
        self.locations = None
        self.track_info = None

    def setup(self):

        self._read_points()
        logger.debug("Locations: %s", self.locations)

        if self.points_spec is not None:

            self.provides(self.points, self.points_spec)
            return

        min_bb = Coordinate(np.floor(np.amin(self.locations, 0)))
        max_bb = Coordinate(np.ceil(np.amax(self.locations, 0)) + 1)

        roi = Roi(min_bb, max_bb - min_bb)

        self.provides(self.points, GraphSpec(roi=roi))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        min_bb = request[self.points].roi.get_begin()
        max_bb = request[self.points].roi.get_end()

        logger.debug(
            "CSV points source got request for %s",
            request[self.points].roi)

        point_filter = np.ones((self.locations.shape[0],), dtype=bool)
        for d in range(self.locations.shape[1]):
            point_filter = np.logical_and(point_filter,
                                          self.locations[:, d] >= min_bb[d])
            point_filter = np.logical_and(point_filter,
                                          self.locations[:, d] < max_bb[d])

        points_data = self._get_points(point_filter)
        logger.debug("Points data: %s", points_data)
        points_spec = GraphSpec(roi=request[self.points].roi.copy())

        batch = Batch()
        batch.graphs[self.points] = Graph(points_data, [], points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_points(self, point_filter):

        filtered_locations = self.locations[point_filter]
        filtered_track_info = self.track_info[point_filter]

        nodes = []
        for location, track_info in zip(filtered_locations,
                                        filtered_track_info):
            node = TrackNode(
                # point_id
                track_info["cell_id"],
                location,
                # parent_id
                track_info["parent_id"]
                if track_info["parent_id"] > 0 else None,
                # track_id
                track_info["track_id"],
                # optional attributes, e.g. radius
                value=track_info["attrs"])
            nodes.append(node)
        return nodes

    def _read_points(self):
        roi = self.points_spec.roi if self.points_spec is not None else None
        self.locations, self.track_info = parse_tracks_file_for_tracks_source(
            self.filename,
            scale=self.scale,
            limit_to_roi=roi,
            attr_filter=self.attr_filter)
        for location, track_info in zip(self.locations,
                                        self.track_info):
            track_info["attrs"]["radius"] = self._set_point_radius(location,
                                                                   track_info)

    def _set_point_radius(self, location, track_info):
        t = location[0]
        radius = None
        if not isinstance(self.use_radius, dict):
            # if use_radius is boolean, take radius from file if set
            if self.use_radius is True:
                radius = track_info["attrs"].get("radius")
        else:
            # otherwise use_radius should be a dict mapping from
            # frame thresholds to radii
            if len(self.use_radius.keys()) > 1:
                for th in sorted(self.use_radius.keys()):
                    # find entry that is closest but larger than
                    # frame of current point
                    if t < int(th):
                        radius = self.use_radius[th]
                        break
                assert radius is not None, \
                    "verify value of use_radius in config"
            else:
                radius = list(self.use_radius.values())[0]
        return radius
