from gunpowder import (Point, Coordinate, Batch, BatchProvider,
                       Roi, PointsSpec, Points)
from gunpowder.profiling import Timing
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrackPoint(Point):

    def __init__(self, location, parent_id, track_id):

        super(TrackPoint, self).__init__(location)

        self.thaw()
        self.original_location = np.array(location, dtype=np.float32)
        self.parent_id = parent_id
        self.track_id = track_id
        self.freeze()


class TracksSource(BatchProvider):
    '''Read tracks of points from a comma-separated-values text file. Each line
    in the file represents one point as::

        [coordinates], point_id, parent_id, track_id

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

    def __init__(self, filename, points, points_spec=None, scale=None):

        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        self.ndims = None
        self.locations = None
        self.track_info = None

    def setup(self):

        self._read_points()

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
        for d in range(self.ndims):
            point_filter = np.logical_and(point_filter,
                                          self.locations[:, d] >= min_bb[d])
            point_filter = np.logical_and(point_filter,
                                          self.locations[:, d] < max_bb[d])

        points_data = self._get_points(point_filter)
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
                track_info[2])
            for location, track_info in zip(filtered_locations,
                                            filtered_track_info)
        }

    def _read_points(self):
        self.locations, self.track_info, self.ndims = self._parse_csv(ndims=-3)

    def _parse_csv(self, ndims=0):
        '''Read one point per line. If ``ndims`` is 0, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        '''
        tokens = [[t.strip(',') for t in line.split()]
                  for line in open(self.filename, 'r')]
        locations = np.array(
            [
                [float(d) for d in line[:ndims]]
                for line in tokens
            ], dtype=np.float32)

        track_info = np.array(
            [
                [int(i.split(".")[0]) for i in line[ndims:]]
                for line in tokens
            ], dtype=np.int32)

        ndims = locations.shape[1]

        if self.scale is not None:
            locations *= self.scale

        return locations, track_info, ndims
