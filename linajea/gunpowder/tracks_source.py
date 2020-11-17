from gunpowder import (Point, Coordinate, Batch, BatchProvider,
                       Roi, PointsSpec, Points)
from gunpowder.profiling import Timing
import numpy as np
import logging

from linajea import parse_tracks_file

logger = logging.getLogger(__name__)


class TrackPoint(Point):

    def __init__(self, location, parent_id, track_id, value=None):

        super(TrackPoint, self).__init__(location)

        self.thaw()
        self.original_location = np.array(location, dtype=np.float32)
        self.parent_id = parent_id
        self.track_id = track_id
        self.value = value
        self.freeze()


class TracksSource(BatchProvider):
    '''Read tracks of points from a comma-separated-values text file.

    If possible, this node uses the header of the file to determine values.
    If present, a header must have the following required fields:
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

    If there is no header, it is assumed that the points are represented
    with values in the following order:

        t, z, y, x, point_id, parent_id, track_id, <radius>, <other values>

    where ``parent_id`` can be -1 to indicate no parent.

    Args:

        filename (``string``):

            The file to read from.

        points (:class:`PointsKey`):

            The key of the points set to create.

        points_spec (:class:`PointsSpec`, optional):

            An optional :class:`PointsSpec` to overwrite the points specs
            automatically determined from the CSV file. This is useful to set
            the :class:`Roi` manually, for example.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the points read
            from the CSV file. This is useful if the points refer to voxel
            positions to convert them to world units.
    '''

    def __init__(self, filename, points, points_spec=None, scale=1.0):

        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
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

        self.provides(self.points, PointsSpec(roi=roi))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        min_bb = request[self.points].roi.get_begin()
        max_bb = request[self.points].roi.get_end()

        logger.debug(
            "CSV points source got request for %s",
            request[self.points].roi)

        point_filter = np.ones((self.locations.shape[0],), dtype=np.bool)
        for d in range(self.locations.shape[1]):
            point_filter = np.logical_and(point_filter,
                                          self.locations[:, d] >= min_bb[d])
            point_filter = np.logical_and(point_filter,
                                          self.locations[:, d] < max_bb[d])

        points_data = self._get_points(point_filter)
        logger.debug("Points data: %s", points_data)
        logger.debug("Type of point: %s", type(list(points_data.values())[0]))
        points_spec = PointsSpec(roi=request[self.points].roi.copy())

        batch = Batch()
        batch.points[self.points] = Points(points_data, points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_points(self, point_filter):

        filtered_locations = self.locations[point_filter]
        filtered_track_info = self.track_info[point_filter]

        return {
            # point_id
            track_info[0]: TrackPoint(
                location,
                # parent_id
                track_info[1] if track_info[1] > 0 else None,
                # track_id
                track_info[2],
                # radius
                value=track_info[3] if len(track_info) > 3 else None)
            for location, track_info in zip(filtered_locations,
                                            filtered_track_info)
        }

    def _read_points(self):
        roi = self.points_spec.roi if self.points_spec is not None else None
        self.locations, self.track_info = parse_tracks_file(
            self.filename,
            scale=self.scale,
            limit_to_roi=roi)
