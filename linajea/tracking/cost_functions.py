"""Provides a set of cost functions for use in solver

Each cost function should return a pair (feature, weight) that, multiplied
togther, results in the cost.
The feature can be constant (typically 1), then it is the same for all
indicators of the respective type, or it can be variable, e.g. a score per
node or edge candidate.

get_default_node_indicator_costs and get_default_edge_indicator_costs can be
used to get the costs for all indicators for the basic and the cell state
setup.
User-provided functions should have the interface:
fn(params: SolveParametersConfig, graph: TrackGraph) ->
    fn_map: {dict str: list of Callable}
params: ILP weights that should be used
graph: Graph that the ILP will be solved on
fn_map: Map that should contain one entry per type of indicator. Each indicator
 can have multiple cost functions associated to it.
 {"indicator_name": [list of cost functions], ...}
 See get_default_node_indicator_costs for an example of such a map.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def feature_times_weight_costs_fn(weight, key="score",
                                  feature_func=lambda x: x):

    def fn(obj):
        feature = feature_func(obj[key])
        return feature, weight

    return fn


def constant_costs_fn(weight, zero_if_true=lambda _: False):

    def fn(obj):
        feature = 1
        cond_weight = 0 if zero_if_true(obj) else weight
        return feature, cond_weight

    return fn


def is_nth_frame(n, frame_key='t'):

    def is_frame(obj):
        return obj[frame_key] == n

    return is_frame


def is_close_to_roi_border(roi, distance):

    def is_close(obj):
        '''Return true if obj is within distance to the z,y,x edge
        of the roi. Assumes 4D data with t,z,y,x'''
        if isinstance(distance, dict):
            dist = min(distance.values())
        else:
            dist = distance

        begin = roi.get_begin()[1:]
        end = roi.get_end()[1:]
        for index, dim in enumerate(['z', 'y', 'x']):
            node_dim = obj[dim]
            begin_dim = begin[index]
            end_dim = end[index]
            if node_dim + dist >= end_dim or\
               node_dim - dist < begin_dim:
                logger.debug("Obj %s with value %s in dimension %s "
                             "is within %s of range [%d, %d]",
                             obj, node_dim, dim, dist,
                             begin_dim, end_dim)
                return True
        logger.debug("Obj %s is not within %s to edge of roi %s",
                     obj, dist, roi)
        return False

    return is_close


def get_default_node_indicator_costs(config, parameters, graph):
    """Get a predefined map of node indicator costs functions

    Args
    ----
    config: TrackingConfig
        Configuration object used, should contain information on which solver
        type to use.
    parameters: SolveParametersConfig
        Current set of weights and parameters used to compute costs.
    graph: TrackGraph
        Graph containing the node candidates for which the costs will be
        computed.
    """
    if parameters.feature_func == "noop":
        feature_func = lambda x: x  # noqa: E731
    elif parameters.feature_func == "log":
        feature_func = np.log
    elif parameters.feature_func == "square":
        feature_func = np.square
    else:
        raise RuntimeError("unknown (non-linear) feature function: %s",
                           parameters.feature_func)

    solver_type = config.solve.solver_type
    fn_map = {
        "node_selected": [
            feature_times_weight_costs_fn(
                parameters.weight_node_score,
                key="score", feature_func=feature_func),
            constant_costs_fn(parameters.selection_constant)],
        "node_appear": [
            constant_costs_fn(
                parameters.track_cost,
                zero_if_true=lambda obj: (
                    is_nth_frame(graph.begin)(obj) or
                    (config.solve.check_node_close_to_roi and
                     is_close_to_roi_border(
                         graph.roi, parameters.max_cell_move)(obj))))]
    }
    if solver_type == "basic":
        fn_map["node_split"] = [
            constant_costs_fn(1)]
    elif solver_type == "cell_state":
        fn_map["node_split"] = [
            feature_times_weight_costs_fn(
                parameters.weight_division,
                key="score_mother", feature_func=feature_func),
            constant_costs_fn(parameters.division_constant)]
        fn_map["node_child"] = [
            feature_times_weight_costs_fn(
                parameters.weight_child,
                key="score_daughter", feature_func=feature_func)]
        fn_map["node_continuation"] = [
            feature_times_weight_costs_fn(
                parameters.weight_continuation,
                key="score_continuation", feature_func=feature_func)]
    else:
        logger.info("solver_type %s unknown for node indicators, skipping",
                    solver_type)

    return fn_map


def get_default_edge_indicator_costs(config, parameters):
    """Get a predefined map of edge indicator costs functions

    Args
    ----
    config: TrackingConfig
        Configuration object used, should contain information on which solver
        type to use.
    parameters: SolveParametersConfig
        Current set of weights and parameters used to compute costs.
    graph: TrackGraph
        Graph containing the node candidates for which the costs will be
        computed (not used for the default edge costs).
    """
    if parameters.feature_func == "noop":
        feature_func = lambda x: x  # noqa: E731
    elif parameters.feature_func == "log":
        feature_func = np.log
    elif parameters.feature_func == "square":
        feature_func = np.square
    else:
        raise RuntimeError("unknown (non-linear) feature function: %s",
                           parameters.feature_func)

    solver_type = config.solve.solver_type
    fn_map = {
        "edge_selected": [
            feature_times_weight_costs_fn(parameters.weight_edge_score,
                                          key="prediction_distance",
                                          feature_func=feature_func)]
    }
    if solver_type == "basic":
        pass
    else:
        logger.info("solver_type %s unknown for edge indicators, skipping",
                    solver_type)

    return fn_map
