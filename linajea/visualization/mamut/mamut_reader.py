import abc

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class MamutReader(ABC):
    def __init__(self):
        super(MamutReader, self).__init__()

    @abc.abstractmethod
    def read_data(self, data):
        # returns a tuple with (cells, edges)
        pass

    def create_cell(self, position, score, _id):
        return {'position': position,
                'score': score,
                'id': _id}

    def create_edge(self, source, target, score):
        return {'source': source,
                'target': target,
                'score': score}

    def create_track(self, start, stop, num_cells, _id, edges):
        return {'start': start,
                'stop': stop,
                'num_cells': num_cells,
                'id': _id,
                'edges': edges}
