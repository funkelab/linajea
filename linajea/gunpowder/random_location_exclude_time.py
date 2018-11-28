import gunpowder as gp


class RandomLocationExcludeTime(gp.RandomLocation):
    ''' time interval is like an array slice - includes start, excludes end '''

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
        self.t_start = time_interval[0]
        self.t_end = time_interval[1]

    def accepts(self, request):

        return (request[self.raw].roi.get_begin()[0] >= self.t_end or
                request[self.raw].roi.get_end()[0] < self.t_start)
