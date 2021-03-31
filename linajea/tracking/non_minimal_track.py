from __future__ import absolute_import
from .non_minimal_solver import NMSolver
from .track_graph import TrackGraph
import logging
import time

logger = logging.getLogger(__name__)


def nm_track(graph, parameters, selected_key, frame_key='t', frames=None):
    ''' A wrapper function that takes a daisy subgraph and input parameters,
    creates and solves the ILP to create tracks, and updates the daisy subgraph
    to reflect the selected nodes and edges.

    Args:

        graph (``daisy.SharedSubgraph``):

            The candidate graph to extract tracks from

        parameters (``TrackingParameters``)

            The parameters to use when optimizing the tracking ILP

        selected_key (``string``)

            The key used to store the `true` or `false` selection status of
            each node and edge in graph.

        frame_key (``string``, optional):

            The name of the node attribute that corresponds to the frame of the
            node. Defaults to "t".

        frames (``list`` of ``int``):

            The start and end frames to solve in (in case the graph doesn't
            have nodes in all frames). Start is inclusive, end is exclusive.
            Defaults to graph.begin, graph.end

    '''
    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        return

    logger.info("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)

    logger.info("Creating solver...")
    solver = NMSolver(track_graph, parameters, selected_key, frames=frames)

    logger.info("Solving...")
    start_time = time.time()
    solver.solve()
    logger.info("Solving ILP took %s seconds" % (time.time() - start_time))

    for cell, data in graph.nodes(data=True):
        data[selected_key] = track_graph.nodes[cell][selected_key]
    for u, v, data in graph.edges(data=True):
        if (u, v) in track_graph.edges:
            data[selected_key] = track_graph.edges[(u, v)][selected_key]
