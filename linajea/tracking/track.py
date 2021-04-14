from __future__ import absolute_import
from .solver import Solver
from .track_graph import TrackGraph
import logging
import time

logger = logging.getLogger(__name__)


def track(graph, parameters, selected_key,
          frame_key='t', frames=None, cell_cycle_key=None):
    ''' A wrapper function that takes a daisy subgraph and input parameters,
    creates and solves the ILP to create tracks, and updates the daisy subgraph
    to reflect the selected nodes and edges.

    Args:

        graph (``daisy.SharedSubgraph``):

            The candidate graph to extract tracks from

        parameters (``TrackingParameters``)

            The parameters to use when optimizing the tracking ILP.
            Can also be a list of parameters.

        selected_key (``string``)

            The key used to store the `true` or `false` selection status of
            each node and edge in graph. Can also be a list of keys
            corresponding to the list of parameters.

        frame_key (``string``, optional):

            The name of the node attribute that corresponds to the frame of the
            node. Defaults to "t".

        frames (``list`` of ``int``):

            The start and end frames to solve in (in case the graph doesn't
            have nodes in all frames). Start is inclusive, end is exclusive.
            Defaults to graph.begin, graph.end

        cell_cycle_key (``string``, optional):

            The name of the node attribute that corresponds to a prediction
            about the cell cycle state. The prediction should be a list of
            three values [mother/division, daughter, continuation].
    '''
    if cell_cycle_key is not None:
        # remove nodes that don't have a cell cycle key, with warning
        to_remove = []
        for node, data in graph.nodes(data=True):
            if cell_cycle_key not in data:
                logger.warning("Node %d does not have cell cycle key %s",
                               node, cell_cycle_key)
                to_remove.append(node)

        for node in to_remove:
            logger.debug("Removing node %d", node)
            graph.remove_node(node)

    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        logger.info("No nodes in graph - skipping solving step")
        return

    if not isinstance(parameters, list):
        parameters = [parameters]
        selected_key = [selected_key]

    assert len(parameters) == len(selected_key),\
        "%d parameter sets and %d selected keys" %\
        (len(parameters), len(selected_key))

    logger.debug("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)

    logger.debug("Creating solver...")
    solver = None
    total_solve_time = 0
    for parameter, key in zip(parameters, selected_key):
        if not solver:
            solver = Solver(track_graph, parameter, key, frames=frames,
                            vgg_key=cell_cycle_key)
        else:
            solver.update_objective(parameter, key)

        logger.debug("Solving for key %s", str(key))
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        total_solve_time += end_time - start_time
        logger.info("Solving ILP took %s seconds", str(end_time - start_time))

        for u, v, data in graph.edges(data=True):
            if (u, v) in track_graph.edges:
                data[key] = track_graph.edges[(u, v)][key]
    logger.info("Solving ILP for all parameters took %s seconds",
                str(total_solve_time))
