"""Provides a function to compute the tracking solution for multiple
parameter sets
"""
import logging
import time

from .solver import Solver
from .track_graph import TrackGraph
from .cost_functions import (get_default_edge_indicator_costs,
                             get_default_node_indicator_costs)
from linajea.tracking import constraints


logger = logging.getLogger(__name__)


def track(graph, config, selected_key, frame_key='t',
          node_indicator_costs=None, edge_indicator_costs=None,
          constraints_fns=[], pin_constraints_fns=[],
          return_solver=False):
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

        node_indicator_costs (Callable):

            Callable that returns a dict of str: list of Callable.
            Will be called once per set of parameters.
            See cost_functions.py:get_default_node_indicator_costs for an
            example.
            The dict should have one entry per type of node indicator.
            The value of each entry is a list of Callables. Each Callable will
            be called on each node and should return a cost. The sum of costs
            is the cost for this indicator in the objective.

        edge_indicator_costs (Callable):

            Callable that returns a dict of str: list of Callable.
            Will be called once per set of parameters.
            See cost_functions.py:get_default_edge_indicator_costs for an
            example.
            The dict should have one entry per type of edge indicator.
            The value of each entry is a list of Callables. Each Callable will
            be called on each edge and should return a cost. The sum of costs
            is the cost for this indicator in the objective.

        constraints_fns (list of Callable)

            Each Callable should handle a single type of constraint.
            It should create the respective constraints for all affected
            objects in the graph and return them.
            Add more Callable to this list to add additional constraints.
            See tracking/constraints.py for examples.
            Interface: fn(graph, indicators) -> constraints
              graph: Create constraints for nodes/edges in this graph
              indicators: The indicator dict created by this Solver object
              constraints: list of pylp.LinearConstraint

        pin_constraints_fns (list of Callable)

            Each Callable should handle a single type of pin constraint.
            Use this to add constraints to pin indicators to specific states.
            Created only if indicator has already been set by neighboring
            blocks.
            Interface: fn(graph, indicators, selected) -> constraints
              graph: Create constraints for nodes/edges in this graph
              indicators: The indicator dict created by this Solver object
              selected_key: Consider this property to determine state of
                candidate
              constraints: list of pylp.LinearConstraint

        return_solver (boolean)

            If True the solver object is returned instead of solving directly.
    '''
    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        logger.info("No nodes in graph - skipping solving step")
        return

    parameters_sets = config.solve.parameters
    if not isinstance(selected_key, list):
        selected_key = [selected_key]

    assert len(parameters_sets) == len(selected_key),\
        "%d parameter sets and %d selected keys" %\
        (len(parameters_sets), len(selected_key))

    logger.debug("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)

    logger.debug("Creating solver...")
    solver = None
    total_solve_time = 0

    if config.solve.solver_type is not None:
        assert (not pin_constraints_fns and
                not constraints_fns), (
                    "Either set solve.solver_type or "
                    "explicitly provide lists of constraint functions")
        constrs = constraints.get_default_constraints(config)
        constraints_fns = constrs[0]
        pin_constraints_fns = constrs[1]

    for parameters, key in zip(parameters_sets, selected_key):
        # set costs depending on current set of parameters
        if config.solve.solver_type is not None:
            assert (not node_indicator_costs and
                    not edge_indicator_costs), (
                        "Either set solve.solver_type or "
                        "explicitly provide cost functions")
            _node_indicator_costs = get_default_node_indicator_costs(
                config, parameters, track_graph)
            _edge_indicator_costs = get_default_edge_indicator_costs(
                config, parameters)
        else:
            assert (node_indicator_costs and
                    edge_indicator_costs), (
                        "Either set solve.solver_type or "
                        "explicitly provide cost functions")
            _node_indicator_costs = node_indicator_costs(parameters,
                                                         track_graph)
            _edge_indicator_costs = edge_indicator_costs(parameters,
                                                         track_graph)

        if not solver:
            solver = Solver(
                track_graph,
                list(_node_indicator_costs.keys()),
                list(_edge_indicator_costs.keys()),
                constraints_fns, pin_constraints_fns,
                timeout=config.solve.timeout)

        solver.update_objective(_node_indicator_costs,
                                _edge_indicator_costs, key)

        if return_solver:
            return solver

        logger.info("Solving for key %s", str(key))
        start_time = time.time()
        solver.solve_and_set()
        end_time = time.time()
        total_solve_time += end_time - start_time
        logger.info("Solving ILP took %s seconds", str(end_time - start_time))

        for u, v, data in graph.edges(data=True):
            if (u, v) in track_graph.edges:
                data[key] = track_graph.edges[(u, v)][key]
    logger.info("Solving ILP for all parameters took %s seconds",
                str(total_solve_time))
