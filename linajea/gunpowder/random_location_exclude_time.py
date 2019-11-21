import gunpowder as gp


class RandomLocationExcludeTime(gp.RandomLocation):
    ''' Provide list of time intervals to exclude.
    time interval is like an array slice - includes start, excludes end '''

    def __init__(
            self,
            raw,
            time_interval,
            min_masked=0,
            mask=None,
            ensure_nonempty=None,
            p_nonempty=1.0):

        super(RandomLocationExcludeTime, self).__init__(
            min_masked,
            mask,
            ensure_nonempty,
            p_nonempty)

        self.raw = raw
        if type(time_interval) is list and \
            (type(time_interval[0]) is list or
                type(time_interval[0] is tuple)):
            self.time_intervals = time_interval
        else:
            self.time_intervals = [time_interval]
        self.t_start = time_interval[0]
        self.t_end = time_interval[1]

    def accepts(self, request):
        for interval in self.time_intervals:
            t_start = interval[0]
            t_end = interval[1]
            if not (request[self.raw].roi.get_begin()[0] >= t_end or
                    request[self.raw].roi.get_end()[0] < t_start):
                return False
        return True
