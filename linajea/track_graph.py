import networkx as nx

class TrackGraph(nx.DiGraph):

    def __init__(self):

        super(TrackGraph, self).__init__(directed=False)

        # self.vp.frame = self.new_vertex_property('int')
        # self.vp.score = self.new_vertex_property('float')
        # self.vp.selected = self.new_vertex_property('bool')
        # self.ep.score = self.new_edge_property('float')
        # self.ep.distance = self.new_edge_property('float')
        # self.ep.selected = self.new_edge_property('bool')
        self.begin = None
        self.end = None
        self._cells_by_frame = {}

    def add_cell(self, cell):

        cell = dict(cell)
        cell_id = cell['id']
        t = cell['position'][0]

        del cell['id']
        cell['frame'] = t
        self.add_node(cell_id, **cell)

        self.begin = t if self.begin is None else min(self.begin, t)
        self.end = t + 1 if self.end is None else max(self.end, t + 1)

        if t not in self._cells_by_frame:
            self._cells_by_frame[t] = []
        self._cells_by_frame[t].append(cell_id)

    def add_cell_edge(self, edge):

        edge = dict(edge)
        source, target = edge['source'], edge['target']

        assert self.nodes[source]['frame'] == self.nodes[target]['frame'] + 1, (
            "Edges are assumed to point backwards in time")

        del edge['source']
        del edge['target']
        self.add_edge(source, target, **edge)

    def prev_edges(self, node):

        return self.out_edges(node)

    def next_edges(self, node):

        return self.in_edges(node)

    def cells_by_frame(self, t):

        print("Getting cells in frame %d"%t)

        if t not in self._cells_by_frame:
            return []
        return self._cells_by_frame[t]
