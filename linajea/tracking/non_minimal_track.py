from __future__ import absolute_import
from .non_minimal_solver import NMSolver
from .track_graph import TrackGraph
import logging
import time

logger = logging.getLogger(__name__)


def nm_track(graph, config, selected_key, frame_key='t', frames=None):
    ''' A wrapper function that takes a daisy subgraph and input parameters,
    creates and solves the ILP to create tracks, and updates the daisy subgraph
    to reflect the selected nodes and edges.

    Args:

        graph (``daisy.SharedSubgraph``):

            The candidate graph to extract tracks from

        config (``TrackingConfig``)

            Configuration object to be used. The parameters to use when
            optimizing the tracking ILP are at config.solve.parameters
            (can also be a list of parameters).

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

    '''
    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        return

    use_cell_state = [p.use_cell_state + "mother"
                      if p.use_cell_state is not None
                      else None
                      for p in config.solve.parameters]
    if any(use_cell_state):
        assert None not in use_cell_state,
        ("mixture of with and without use_cell_state in concurrent "
         "solving not supported yet")

    parameters = config.solve.parameters
    if not isinstance(selected_key, list):
        selected_key = [selected_key]

    assert len(parameters) == len(selected_key),\
        "%d parameter sets and %d selected keys" %\
        (len(parameters), len(selected_key))

    logger.debug("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)

    logger.info("Creating solver...")
    solver = None
    total_solve_time = 0
    for parameter, key in zip(parameters, selected_key):
        if not solver:
            solver = NMSolver(
                track_graph, parameter, key, frames=frames,
                check_node_close_to_roi=config.solve.check_node_close_to_roi,
                timeout=config.solve.timeout,
                add_node_density_constraints=config.solve.add_node_density_constraints)
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
