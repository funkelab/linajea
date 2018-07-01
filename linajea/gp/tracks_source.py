from gunpowder import CsvPointsSource, Point
import numpy as np

class TrackPoint(Point):

    def __init__(self, location, parent_id, track_id):

        super(TrackPoint, self).__init__(location)

        self.thaw()
        self.original_location = np.array(location, dtype=np.float32)
        self.parent_id = parent_id
        self.track_id = track_id
        self.freeze()

class TracksSource(CsvPointsSource):
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

    def _get_points(self, point_filter):

        filtered = self.data[point_filter]

        return {
            row[self.ndims]: TrackPoint(
                row[:self.ndims], # location
                row[self.ndims + 1] if row[self.ndims + 1] >= 0 else None, # parent id
                row[self.ndims + 2] # track id
            )
            for row in filtered
        }

    def _read_points(self):
        # we expect 3 non-location entries per row
        self.data, self.ndims = self._parse_csv(-3)
