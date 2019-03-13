import networkx as nx
import logging

logger = logging.getLogger(__name__)


class TrackGraph(nx.DiGraph):
    '''A track graph of cells and inter-frame edges between them.

    Args:

        graph_data (optional):

            Optional graph data to pass to the networkx.Graph constructor as
            ``incoming_graph_data``. This can be used to populate a track graph
            with entries from a generic networkx graph. For use in solver, this
            must be a candidate graph, or something with the .roi attribute.

        frame_key (``string``, optional):

            The name of the node attribute that corresponds to the frame of the
            node. Defaults to "frame".

    '''

    def __init__(
            self,
            graph_data=None,
            frame_key='frame'):

        super(TrackGraph, self).__init__(incoming_graph_data=graph_data)

        self.begin = None
        self.end = None
        self._cells_by_frame = {}
        self.frame_key = frame_key
        self.roi = graph_data.roi

        if graph_data is not None:
            frames = [
                self.nodes[cell][self.frame_key]
                for cell in self.nodes
            ]
            self.begin = min(frames)
            self.end = max(frames) + 1
            for cell in self.nodes:
                t = self.nodes[cell][self.frame_key]
                if t not in self._cells_by_frame:
                    self._cells_by_frame[t] = []
                self._cells_by_frame[t].append(cell)

        for u, v in self.edges:
            if (
                    self.nodes[u][self.frame_key] <=
                    self.nodes[v][self.frame_key]):
                raise RuntimeError(
                    "edge from %d to %d does not go backwards in time, but "
                    "from frame %d to %d" % (
                        u, v,
                        self.nodes[u][self.frame_key],
                        self.nodes[v][self.frame_key]))

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

    def get_tracks(self, require_selected=False, selected_key='selected'):
        '''Get a generator of track graphs, each corresponding to one track
        (i.e., a connected component in the track graph).

        Args:

            require_selected (``bool``):

                If ``True``, consider only edges that have a selected_key
                attribute that is set to ``True``. Otherwise, each edge will be
                considered for the connected component analysis.

            selected_key (``str``):

                Only used if require_selected=True. Determines the attribute
                name to check if an edge is selected. Default value is
                'selected'.

        Returns:

            A generator object of graphs, one for each track.
        '''

        if not require_selected:

            graph = self

        else:

            selected_edges = [
                e
                for e in self.edges
                if (selected_key in self.edges[e]
                    and self.edges[e][selected_key])
            ]
            graph = self.edge_subgraph(selected_edges)

        return [
            TrackGraph(graph_data=graph.subgraph(g).copy())
            for g in nx.weakly_connected_components(graph)
        ]
