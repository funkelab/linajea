"""Provides a solver object encapsulating the ILP solver
"""
# -*- coding: utf-8 -*-
import logging

import pylp

from .constraints import (ensure_at_most_two_successors,
                          ensure_edge_endpoints,
                          ensure_one_predecessor,
                          ensure_pinned_edge,
                          ensure_split_set_for_divs,
                          ensure_one_state,
                          ensure_split_child)

logger = logging.getLogger(__name__)


class Solver(object):
    '''
    Class for initializing and solving the ILP problem for
    creating tracks from candidate nodes and edges using pylp.
    '''
    def __init__(self, track_graph,
                 node_indicator_keys, edge_indicator_keys,
                 pin_constraints_fn_list, edge_constraints_fn_list,
                 node_constraints_fn_list, inter_frame_constraints_fn_list,
                 timeout=120):

        self.graph = track_graph
        self.timeout = timeout

        self.pinned_edges = {}

        self.num_vars = None
        self.objective = None
        self.main_constraints = []  # list of LinearConstraint objects
        self.pin_constraints = []  # list of LinearConstraint objects
        self.solver = None

        self.node_indicator_keys = set(node_indicator_keys)
        self.edge_indicator_keys = set(edge_indicator_keys)

        self.pin_constraints_fn_list = pin_constraints_fn_list
        self.edge_constraints_fn_list = edge_constraints_fn_list
        self.node_constraints_fn_list = node_constraints_fn_list
        self.inter_frame_constraints_fn_list = inter_frame_constraints_fn_list

        self._create_indicators()
        self._create_solver()
        self._create_constraints()

    def update_objective(self, node_indicator_fn_map, edge_indicator_fn_map,
                         selected_key):
        self.node_indicator_fn_map = node_indicator_fn_map
        self.edge_indicator_fn_map = edge_indicator_fn_map
        assert (set(self.node_indicator_fn_map.keys()) == self.node_indicator_keys and
                set(self.edge_indicator_fn_map.keys()) == self.edge_indicator_keys), \
            "cannot change set of indicators during one run!"

        self.selected_key = selected_key

        self._set_objective()
        self.solver.set_objective(self.objective)

        self.pinned_edges = {}
        self.pin_constraints = []
        self._add_pin_constraints()
        all_constraints = pylp.LinearConstraints()
        for c in self.main_constraints + self.pin_constraints:
            all_constraints.add(c)
        self.solver.set_constraints(all_constraints)

    def _create_solver(self):
        self.solver = pylp.LinearSolver(
                self.num_vars,
                pylp.VariableType.Binary,
                preference=pylp.Preference.Any)
        self.solver.set_num_threads(1)
        self.solver.set_timeout(self.timeout)

    def solve(self):
        solution, message = self.solver.solve()
        logger.info(message)
        logger.info("costs of solution: %f", solution.get_value())

        return solution

    def solve_and_set(self, node_key="node_selected", edge_key="edge_selected"):
        solution = self.solve()

        for v in self.graph.nodes:
            self.graph.nodes[v][self.selected_key] = solution[
                self.indicators[node_key][v]] > 0.5

        for e in self.graph.edges:
            self.graph.edges[e][self.selected_key] = solution[
                self.indicators[edge_key][e]] > 0.5

    def _create_indicators(self):

        self.indicators = {}
        self.num_vars = 0

        for k in self.node_indicator_keys:
            self.indicators[k] = {}
            for node in self.graph.nodes:
                self.indicators[k][node] = self.num_vars
                self.num_vars += 1

        for k in self.edge_indicator_keys:
            self.indicators[k] = {}
            for edge in self.graph.edges():
                self.indicators[k][edge] = self.num_vars
                self.num_vars += 1

    def _set_objective(self):

        logger.debug("setting objective")

        objective = pylp.LinearObjective(self.num_vars)

        # node costs
        for k, fn in self.node_indicator_fn_map.items():
            for n_id, node in self.graph.nodes(data=True):
                objective.set_coefficient(self.indicators[k][n_id], sum(fn(node)))

        # edge costs
        for k, fn in self.edge_indicator_fn_map.items():
            for u, v, edge in self.graph.edges(data=True):
                objective.set_coefficient(self.indicators[k][(u, v)], sum(fn(edge)))

        self.objective = objective

    def _create_constraints(self):

        self.main_constraints = []

        self._add_edge_constraints()
        self._add_node_constraints()

        for t in range(self.graph.begin, self.graph.end):
            self._add_inter_frame_constraints(t)


    def _add_pin_constraints(self):

        logger.debug("setting pin constraints: %s",
                     self.pin_constraints_fn_list)

        for edge in self.graph.edges():
            if self.selected_key in self.graph.edges[edge]:
                selected = self.graph.edges[edge][self.selected_key]
                self.pinned_edges[edge] = selected

                for fn in self.pin_constraints_fn_list:
                    self.pin_constraints.extend(
                        fn(edge, self.indicators, selected))

    def _add_edge_constraints(self):

        logger.debug("setting edge constraints: %s",
                     self.edge_constraints_fn_list)

        for edge in self.graph.edges():
            for fn in self.edge_constraints_fn_list:
                self.main_constraints.extend(fn(edge, self.indicators))

    def _add_node_constraints(self):

        logger.debug("setting node constraints: %s",
                     self.node_constraints_fn_list)

        for node in self.graph.nodes():
            for fn in self.node_constraints_fn_list:
                self.main_constraints.extend(fn(node, self.indicators))


    def _add_inter_frame_constraints(self, t):
        '''Linking constraints from t to t+1.'''

        logger.debug("setting inter-frame constraints for frame %d: %s", t,
                     self.inter_frame_constraints_fn_list)

        for node in self.graph.cells_by_frame(t):
            for fn in self.inter_frame_constraints_fn_list:
                self.main_constraints.extend(
                    fn(node, self.indicators, self.graph, pinned_edges=self.pinned_edges))


class BasicSolver(Solver):
    '''Specialized class initialized with the basic indicators and constraints
    '''
    def __init__(self, track_graph, timeout=120):

        pin_constraints_fn_list = [ensure_pinned_edge]
        edge_constraints_fn_list = [ensure_edge_endpoints]
        node_constraints_fn_list = []
        inter_frame_constraints_fn_list = [
            ensure_one_predecessor,
            ensure_at_most_two_successors,
            ensure_split_set_for_divs]

        node_indicator_keys = ["node_selected", "node_appear", "node_split"]
        edge_indicator_keys = ["edge_selected"]

        super(BasicSolver, self).__init__(
            track_graph, node_indicator_keys, edge_indicator_keys,
            pin_constraints_fn_list, edge_constraints_fn_list,
            node_constraints_fn_list, inter_frame_constraints_fn_list,
            timeout=timeout)


class CellStateSolver(Solver):
    '''Specialized class initialized with the indicators and constraints
    necessary to include cell state information in addition to the basic
    indicators and constraints
    '''
    def __init__(self, track_graph, timeout=120):

        pin_constraints_fn_list = [ensure_pinned_edge]
        edge_constraints_fn_list = [ensure_edge_endpoints, ensure_split_child]
        node_constraints_fn_list = [ensure_one_state]
        inter_frame_constraints_fn_list = [
            ensure_one_predecessor,
            ensure_at_most_two_successors,
            ensure_split_set_for_divs]

        node_indicator_keys = ["node_selected", "node_appear", "node_split",
                               "node_child", "node_continuation"]
        edge_indicator_keys = ["edge_selected"]

        super(CellStateSolver, self).__init__(
            track_graph, node_indicator_keys, edge_indicator_keys,
            pin_constraints_fn_list, edge_constraints_fn_list,
            node_constraints_fn_list, inter_frame_constraints_fn_list,
            timeout=timeout)
