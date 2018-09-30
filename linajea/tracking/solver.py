# -*- coding: UTF-8 -*-
import logging
import math
import numpy as np
import pylp

logger = logging.getLogger(__name__)

class Solver(object):

    def __init__(self, graph, parameters):

        self.graph = graph
        self.parameters = parameters

        self.node_selected = {}
        self.edge_selected = {}
        self.node_appear = {}
        self.node_disappear = {}
        self.pinned_edges = {}

        self.num_vars = None
        self.objective = None
        self.constraints = None

    def solve(self):

        self._create_indicators()
        self._set_objective()
        self._add_constraints()

        solver = pylp.create_linear_solver(pylp.Preference.Any)
        solver.initialize(self.num_vars, pylp.VariableType.Binary)

        solver.set_objective(self.objective)
        solver.set_constraints(self.constraints)

        solution, message = solver.solve()
        logger.info(message)
        logger.info("costs of solution: %f", solution.get_value())

        for v in self.graph.nodes:
            self.graph.nodes[v]['selected'] = solution[self.node_selected[v]] > 0.5

        for e in self.graph.edges:
            self.graph.edges[e]['selected'] = solution[self.edge_selected[e]] > 0.5

    def _create_indicators(self):

        self.num_vars = 0

        # three indicators per node:
        #   1. selected
        #   2. appear
        #   3. disappear
        for node in self.graph.nodes:
            self.node_selected[node] = self.num_vars
            self.node_appear[node] = self.num_vars + 1
            self.node_disappear[node] = self.num_vars + 2
            self.num_vars += 3

        for edge in self.graph.edges():
            self.edge_selected[edge] = self.num_vars
            self.num_vars += 1

    def _set_objective(self):

        logger.debug("setting objective")

        objective = pylp.LinearObjective(self.num_vars)

        # node selection, appear, and disappear costs
        for node in self.graph.nodes:
            objective.set_coefficient(
                self.node_selected[node],
                self._node_costs(node))
            objective.set_coefficient(
                self.node_appear[node],
                self.parameters.cost_appear)
            objective.set_coefficient(
                self.node_disappear[node],
                self.parameters.cost_disappear)

        # edge selection costs
        for edge in self.graph.edges():
            objective.set_coefficient(
                self.edge_selected[edge],
                self._edge_costs(edge))

        self.objective = objective

    def _node_costs(self, node):

        # simple linear costs based on the score of a node (negative if above
        # threshold_node_score, positive otherwise)

        return self.parameters.threshold_node_score - self.graph.nodes[node]['score']

    def _edge_costs(self, edge):

        # simple linear costs based on the score of an edge (negative if above
        # threshold_edge_score, positive otherwise)

        score_costs = (
            self.parameters.threshold_edge_score -
            self.graph.edges[edge]['score'])

        # plus costs for the distance between the linked nodes

        move_costs = (
            self.graph.edges[edge]['distance']*
            self.parameters.weight_distance_cost)

        return score_costs + move_costs

    def _add_constraints(self):

        self.constraints = pylp.LinearConstraints()

        self._add_pin_constraints()
        self._add_edge_constraints()

        for t in range(self.graph.begin, self.graph.end):
            self._add_inter_frame_constraints(t)

    def _add_pin_constraints(self):

        for e in self.graph.edges():

            if 'selected' in self.graph.edges[e]:

                selected = self.graph.edges[e]['selected']
                self.pinned_edges[e] = selected

                ind_e = self.edge_selected[e]
                constraint = pylp.LinearConstraint()
                constraint.set_coefficient(ind_e, 1)
                constraint.set_relation(pylp.Relation.Equal)
                constraint.set_value(1 if selected else 0)
                self.constraints.add(constraint)

    def _add_edge_constraints(self):

        logger.debug("setting edge constraints")

        for e in self.graph.edges():

            # if e is selected, u and v have to be selected
            u, v = e
            ind_e = self.edge_selected[e]
            ind_u = self.node_selected[u]
            ind_v = self.node_selected[v]

            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(ind_e, 2)
            constraint.set_coefficient(ind_u, -1)
            constraint.set_coefficient(ind_v, -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(0)
            self.constraints.add(constraint)

            logger.debug("set edge constraint %s", constraint)

    def _add_inter_frame_constraints(self, t):
        '''Linking constraints from t to t+1.'''

        logger.debug("setting inter-frame constraints for frame %d", t)

        # Every selected node has exactly one selected edge to the previous and
        # one or two to the next frame. This includes the special "appear" and
        # "disappear" edges.
        for node in self.graph.cells_by_frame(t):

            # we model this as three constraints:
            #  sum(prev) -   node  = 0 # exactly one prev edge, iff node selected
            #  sum(next) - 2*node <= 0 # at most two next edges
            # -sum(next) +   node <= 0 # at least one next, iff node selected

            constraint_prev = pylp.LinearConstraint()
            constraint_next_1 = pylp.LinearConstraint()
            constraint_next_2 = pylp.LinearConstraint()

            # sum(prev)

            # all neighbors in previous frame
            pinned_to_1 = []
            for edge in self.graph.prev_edges(node):
                constraint_prev.set_coefficient(self.edge_selected[edge], 1)
                if edge in self.pinned_edges and self.pinned_edges[edge]:
                    pinned_to_1.append(edge)
            if len(pinned_to_1) > 1:
                raise RuntimeError(
                    "Node %d has more than one prev edge pinned: %s"%(
                        node,
                        pinned_to_1))
            # plus "appear"
            constraint_prev.set_coefficient(self.node_appear[node], 1)

            # sum(next)

            for edge in self.graph.next_edges(node):
                constraint_next_1.set_coefficient(self.edge_selected[edge], 1)
                constraint_next_2.set_coefficient(self.edge_selected[edge], -1)
            # plus "disappear"
            constraint_next_1.set_coefficient(self.node_disappear[node], 1)
            constraint_next_2.set_coefficient(self.node_disappear[node], -1)

            # node

            constraint_prev.set_coefficient(self.node_selected[node], -1)
            constraint_next_1.set_coefficient(self.node_selected[node], -2)
            constraint_next_2.set_coefficient(self.node_selected[node], 1)

            # relation, value

            constraint_prev.set_relation(pylp.Relation.Equal)
            constraint_next_1.set_relation(pylp.Relation.LessEqual)
            constraint_next_2.set_relation(pylp.Relation.LessEqual)

            constraint_prev.set_value(0)
            constraint_next_1.set_value(0)
            constraint_next_2.set_value(0)

            self.constraints.add(constraint_prev)
            self.constraints.add(constraint_next_1)
            self.constraints.add(constraint_next_2)

            logger.debug(
                "set inter-frame constraints:\t%s\n\t%s\n\t%s",
                constraint_prev, constraint_next_1, constraint_next_2)
