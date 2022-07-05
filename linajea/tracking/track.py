"""Provides a function to compute the tracking solution for multiple
parameter sets
"""
import logging
import os
import time

import numpy as np
import pylp

from .solver import Solver
from .track_graph import TrackGraph
from .cost_functions import (get_edge_indicator_fn_map_default,
                             get_node_indicator_fn_map_default)
from linajea.tracking import constraints


logger = logging.getLogger(__name__)


def track(graph, config, selected_key, frame_key='t',
          node_indicator_fn_map=None, edge_indicator_fn_map=None,
          pin_constraints_fn_list=[], edge_constraints_fn_list=[],
          node_constraints_fn_list=[], inter_frame_constraints_fn_list=[],
          block_id=0):
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

        node_indicator_fn_map (Callable):

            Callable that returns a dict of str: Callable.
            One entry per type of node indicator the solver should have;
            The Callable stored in the value of each entry will be called
            on each node and should return the cost for this indicator in
            the objective.

        edge_indicator_fn_map (Callable):

            Callable that returns a dict of str: Callable.
            One entry per type of edge indicator the solver should have;
            The Callable stored in the value of each entry will be called
            on each edge and should return the cost for this indicator in
            the objective.

        pin_constraints_fn_list (list of Callable)

            A list of Callable that return a list of pylp.LinearConstraint each.
            Use this to add constraints to pin edge indicators to specific
            states. Called only if edge has already been set by neighboring
            blocks.
            Interface: fn(edge, indicators, selected)
              edge: Create constraints for this edge
              indicators: The indicator map created by the Solver object
              selected: Will be set to the selected state of this edge

        edge_constraints_fn_list (list of Callable)

            A list of Callable that return a list of pylp.LinearConstraint each.
            Use this to add constraints on a specific edge.
            Interface: fn(edge, indicators)
              edge: Create constraints for this edge
              indicators: The indicator map created by the Solver object

        node_constraints_fn_list (list of Callable)

            A list of Callable that return a list of pylp.LinearConstraint each.
            Use this to add constraints on a specific node.
            Interface: fn(node, indicators)
              node: Create constraints for this node
              indicators: The indicator map created by the Solver object

        inter_frame_constraints_fn_list (list of Callable)

            A list of Callable that return a list of pylp.LinearConstraint each.
            d
            Interface: fn(node, indicators, graph, **kwargs)
              node: Create constraints for this node
              indicators: The indicator map created by the Solver object
              graph: The track graph that is solved.
              **kwwargs: Additional parameters, so far only contains
                `pinned_edges`, requires changes to Solver object to extend.

        block_id (``int``, optional):

            The ID of the current daisy block if data is processed block-wise.
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
        constrs = constraints.get_constraints_default(config)
        pin_constraints_fn_list = constrs[0]
        edge_constraints_fn_list = constrs[1]
        node_constraints_fn_list = constrs[2]
        inter_frame_constraints_fn_list = constrs[3]

    for parameters, key in zip(parameters_sets, selected_key):
        if node_indicator_fn_map is None:
            _node_indicator_fn_map = get_node_indicator_fn_map_default(
                config, parameters, track_graph)
        else:
            _node_indicator_fn_map = node_indicator_fn_map(config, parameters,
                                                           track_graph)
        if edge_indicator_fn_map is None:
            _edge_indicator_fn_map = get_edge_indicator_fn_map_default(
                config, parameters)
        else:
            _edge_indicator_fn_map = edge_indicator_fn_map(config, parameters,
                                                           track_graph)

        if not solver:
            solver = Solver(
                track_graph,
                list(_node_indicator_fn_map.keys()),
                list(_edge_indicator_fn_map.keys()),
                pin_constraints_fn_list, edge_constraints_fn_list,
                node_constraints_fn_list, inter_frame_constraints_fn_list,
                timeout=config.solve.timeout)

        solver.update_objective(_node_indicator_fn_map,
                                _edge_indicator_fn_map, key)

        if config.solve.write_struct_svm:
            write_struct_svm(solver, block_id, config.solve.write_struct_svm)
            logger.info("wrote struct svm data, skipping solving")
            break
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


def write_struct_svm(solver, block_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # write all indicators as "object_id, indicator_id"
    for k, objs in solver.indicators.items():
        with open(os.path.join(output_dir, k + "_b" + str(block_id)), 'w') as f:
            for obj, v in objs.items():
                f.write(f"{obj} {v}\n")

    # write features for objective
    indicators = {}
    num_features = {}
    for k, fn in solver.node_indicator_fn_map.items():
        for n_id, node in solver.graph.nodes(data=True):
            ind = solver.indicators[k][n_id]
            costs = fn(node)
            indicators[ind] = (k, costs)
            num_features[k] = len(costs)
    for k, fn in solver.edge_indicator_fn_map.items():
        for u, v, edge in solver.graph.edges(data=True):
            ind = solver.indicators[k][(u, v)]
            costs = fn(edge)
            indicators[ind] = (k, costs)
            num_features[k] = len(costs)

    features_locs = {}
    acc = 0
    for k, v in num_features.items():
        features_locs[k] = acc
        acc += v
    num_features = acc
    assert sorted(indicators.keys()) == list(range(len(indicators))), \
        "some error reading indicators and features"
    with open(os.path.join(output_dir, "features_b" + str(block_id)), 'w') as f:
        for ind in sorted(indicators.keys()):
            k, costs = indicators[ind]
            features = [0]*num_features
            features_loc = features_locs[k]
            features[features_loc:features_loc+len(costs)] = costs
            f.write(" ".join([str(f) for f in features]) + "\n")

    # write constraints
    def rel_to_str(rel):
        if rel == pylp.Relation.Equal:
            return " == "
        elif rel == pylp.Relation.LessEqual:
            return " <= "
        elif rel == pylp.Relation.GreaterEqual:
            return " >= "
        else:
            raise RuntimeError("invalid pylp.Relation: %s", rel)

    with open(os.path.join(output_dir, "constraints_b" + str(block_id)), 'w') as f:
        for constraint in solver.main_constraints:
            val = constraint.get_value()
            rel = rel_to_str(constraint.get_relation())
            coeffs = " ".join([f"{v}*{idx}"
                               for idx, v in constraint.get_coefficients().items()])
            f.write(f"{coeffs} {rel} {val}\n")
