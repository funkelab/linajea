import networkx as nx
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TrackGraph(nx.DiGraph):
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

        super(TrackGraph, self).__init__(incoming_graph_data=graph_data)

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
                    logger.debug("{} in graph: {} \t {} in graph: {}".format(u, u in self.nodes, v, v in self.nodes))
                    skipped_edges += 1

            logger.info("Skipped %d edges without corresponding nodes", skipped_edges)

    def add_cell(self, cell):
        '''Add a cell as a node to the graph.

        Args:

            cell (``dict``):

                A dictionary containing at least the keys ``id`` and
                ``position``, and does not contain ``frame``. Other keys will be
                added as properties to the node.
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

        assert self.nodes[source]['frame'] > self.nodes[target]['frame'], (
            "Edges are assumed to point backwards in time, but edge (%d, %d) "
            "points from frame %d to %d"%(
                source, target,
                self.nodes[source]['frame'],
                self.nodes[target]['frame']))

        del edge['source']
        del edge['target']
        self.add_edge(source, target, **edge)

    def prev_edges(self, node):
        '''Get all edges that point backward from ``node``.'''

        return self.out_edges(node)

    def next_edges(self, node):
        '''Get all edges that point forward from ``node``.'''

        return self.in_edges(node)

    def get_frames(self):
        '''Get a tuple ``(t_1, t_2)`` of the first and last frame this track
        graph has nodes for.'''

        return (min(self._cells_by_frame.keys()),
                max(self._cells_by_frame.keys()))

    def cells_by_frame(self, t):
        '''Get all cells in frame ``t``.'''

        if t not in self._cells_by_frame:
            return []
        return self._cells_by_frame[t]

    def get_tracks(self, require_selected=False):
        '''Get a generator of track graphs, each corresponding to one track
        (i.e., a connected component in the track graph).

        Args:

            require_selected (``bool``):

                If ``True``, consider only edges that have a ``selected``
                attribute that is set to ``True``. Otherwise, each edge will be
                considered for the connected component analysis.
ls

        Returns:

            A generator object of graphs, one for each track.
        '''

        if not require_selected:

            graph = self

        else:

            selected_edges = [
                e
                for e in self.edges
                if 'selected' in self.edges[e] and self.edges[e]['selected']
            ]
            graph = self.edge_subgraph(selected_edges)

        return [
            TrackGraph(graph_data=graph.subgraph(g).copy())
            for g in nx.weakly_connected_components(graph)
        ]
