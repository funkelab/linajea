"""Provides gunpowder node to exclude time range when selecting
 random location
"""
import gunpowder as gp
import logging

logger = logging.getLogger(__name__)


class RandomLocationExcludeTime(gp.RandomLocation):
    """Adapts Gunpowder RandomLocation node to exclude time interval

    Provide list of time intervals to exclude.
    time interval is like an array slice - includes start, excludes end

    Attributes
    ----------
    raw: ArrayKey
        verify that Roi of this array is outside of all given intervals
    time_interval: list of list of int
        list of intervals to exclude
    min_masked: float
    mask: ArrayKey
    ensure_nonempty: GraphKey
    p_nonempty: float
    point_balance_radius: int
        Pass these through to RandomLocation constructor
    """

    def __init__(
            self,
            raw,
            time_interval,
            min_masked=0,
            mask=None,
            ensure_nonempty=None,
            p_nonempty=1.0,
            point_balance_radius=1):

        super(RandomLocationExcludeTime, self).__init__(
            min_masked,
            mask,
            ensure_nonempty,
            p_nonempty,
            point_balance_radius=point_balance_radius)

        self.raw = raw
        if isinstance(time_interval, list) and \
            (isinstance(time_interval[0], list) or
                isinstance(time_interval[0], tuple)):
            self.time_intervals = time_interval
        else:
            self.time_intervals = [time_interval]

    def accepts(self, request):
        for interval in self.time_intervals:
            logger.debug("Interval: %s" % str(interval))
            t_start = interval[0]
            t_end = interval[1]
            if not (request[self.raw].roi.get_begin()[0] >= t_end or
                    request[self.raw].roi.get_end()[0] < t_start):
                return False
        return True
