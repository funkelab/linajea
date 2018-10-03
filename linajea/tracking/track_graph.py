import networkx as nx
import logging

logger = logging.getLogger(__name__)

class TrackGraph(nx.Graph):
    '''A track graph of cells and inter-frame edges between them.

    Args:

        cells (``dict``, optional):
        edges (``dict``, optional):

            If given, populate the graph with these cells and edges. Edges to
            or from cells that are not in ``cells`` will not be added.

        graph_data (optional):

            Optional graph data to pass to the networkx.Graph constructor as
            ``incoming_graph_data``. This can be used to populate a track graph
            with entries from a generic networkx graph.
    '''

    def __init__(self, cells=None, edges=None, graph_data=None):

        super(TrackGraph, self).__init__(incoming_graph_data=graph_data, directed=False)

        self.begin = None
        self.end = None
        self._cells_by_frame = {}

        if graph_data is not None:
            frames = [ self.nodes[cell]['frame'] for cell in self.nodes ]
            self.begin = min(frames)
            self.end = max(frames) + 1
            for cell in self.nodes:
                t = self.nodes[cell]['frame']
                if t not in self._cells_by_frame:
                    self._cells_by_frame[t] = []
                self._cells_by_frame[t].append(cell)

        if cells is not None:
            for cell in cells:
                self.add_cell(cell)

        if edges is not None:

            skipped_edges = 0
            for edge in edges:

                u, v = edge['source'], edge['target']

                if u in self.nodes and v in self.nodes:

                    self.add_cell_edge(edge)

                else:

                    logger.debug(
                        "Skipping edge %d -> %d, at least one node not in graph",
                        u, v)
                    skipped_edges += 1

            logger.info("Skipped %d edges without corresponding nodes", skipped_edges)

    def add_cell(self, cell):
        '''Add a cell as a node to the graph.

        Args:

            cell (``dict``):

                A dictionary containing at least the keys ``id``, ``position``,
                and ``frame``. Other keys will be added as properties to the
                node.
        '''

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
        '''Add a directed edge between cells.

        Args:

            edge (``dict``):

                A dictionary containing at least the keys ``source`` and
                ``target``, which correspond to cell IDs. The edge has to point
                backwards in time. Other keys will be added as properties to
                the edge.
        '''

        edge = dict(edge)
        source, target = edge['source'], edge['target']

        assert self.nodes[source]['frame'] == self.nodes[target]['frame'] + 1, (
            "Edges are assumed to point backwards in time")

        del edge['source']
        del edge['target']
        self.add_edge(source, target, **edge)

    def prev_edges(self, node):
        '''Get all edges that point backward from ``node``.'''

        return self.out_edges(node)

    def next_edges(self, node):
        '''Get all edges that point forward from ``node``.'''

        return self.in_edges(node)

    def cells_by_frame(self, t):
        '''Get all cells in frame ``t``.'''

        if t not in self._cells_by_frame:
            return []
        return self._cells_by_frame[t]
