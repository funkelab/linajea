# -*- coding: utf-8 -*-
import logging
import os
import time

import numpy as np

import pylp

logger = logging.getLogger(__name__)


class Solver(object):
    '''
    Class for initializing and solving the ILP problem for
    creating tracks from candidate nodes and edges using pylp.
    This is the "minimal" version, simplified to minimize the
    number of hyperparamters
    '''
    def __init__(self, track_graph, parameters, selected_key, frames=None,
                 write_struct_svm=False, block_id=None,
                 check_node_close_to_roi=True, timeout=120,
                 add_node_density_constraints=False):
        # frames: [start_frame, end_frame] where start_frame is inclusive
        # and end_frame is exclusive. Defaults to track_graph.begin,
        # track_graph.end
        self.write_struct_svm = write_struct_svm
        if self.write_struct_svm:
            assert isinstance(self.write_struct_svm, str)
            os.makedirs(self.write_struct_svm, exist_ok=True)
        self.check_node_close_to_roi = check_node_close_to_roi
        self.add_node_density_constraints = add_node_density_constraints
        self.block_id = block_id

        if parameters.feature_func == "noop":
            self.feature_func = lambda x: x
        elif parameters.feature_func == "log":
            self.feature_func = np.log
        elif parameters.feature_func == "square":
            self.feature_func = np.square
        else:
            raise RuntimeError("invalid feature_func parameters %s", parameters.feature_func)

        self.graph = track_graph
        self.start_frame = frames[0] if frames else self.graph.begin
        self.end_frame = frames[1] if frames else self.graph.end
        self.timeout = timeout

        self.node_selected = {}
        self.edge_selected = {}
        self.node_appear = {}
        self.node_disappear = {}
        self.node_split = {}
        self.node_child = {}
        self.node_continuation = {}
        self.pinned_edges = {}

        self.num_vars = None
        self.objective = None
        self.main_constraints = []  # list of LinearConstraint objects
        self.pin_constraints = []  # list of LinearConstraint objects
        self.solver = None

        logger.debug("cell cycle key? %s", parameters.cell_cycle_key)
        logger.debug("write ssvm? %s", self.write_struct_svm)

        self._create_indicators()
        self._create_solver()
        self._create_constraints()
        self.update_objective(parameters, selected_key)

    def update_objective(self, parameters, selected_key):
        self.parameters = parameters
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

        for v in self.graph.nodes:
            self.graph.nodes[v][self.selected_key] = solution[
                    self.node_selected[v]] > 0.5

        for e in self.graph.edges:
            self.graph.edges[e][self.selected_key] = solution[
                    self.edge_selected[e]] > 0.5

    def _create_indicators(self):

        self.num_vars = 0

        # four indicators per node:
        #   1. selected
        #   2. appear
        #   3. disappear
        #   4. split
        if self.write_struct_svm:
            node_selected_file = open(f"{self.write_struct_svm}/node_selected_b{self.block_id}", 'w')
            node_appear_file = open(f"{self.write_struct_svm}/node_appear_b{self.block_id}", 'w')
            node_disappear_file = open(f"{self.write_struct_svm}/node_disappear_b{self.block_id}", 'w')
            node_split_file = open(f"{self.write_struct_svm}/node_split_b{self.block_id}", 'w')
            node_child_file = open(f"{self.write_struct_svm}/node_child_b{self.block_id}", 'w')
            node_continuation_file = open(f"{self.write_struct_svm}/node_continuation_b{self.block_id}", 'w')
        else:
            node_selected_file = None
            node_appear_file = None
            node_disappear_file = None
            node_split_file = None
            node_child_file = None
            node_continuation_file = None
        for node in self.graph.nodes:
            if self.write_struct_svm:
                node_selected_file.write("{} {}\n".format(node, self.num_vars))
                node_appear_file.write("{} {}\n".format(node, self.num_vars + 1))
                node_disappear_file.write("{} {}\n".format(node, self.num_vars + 2))
                node_split_file.write("{} {}\n".format(node, self.num_vars + 3))
                node_child_file.write("{} {}\n".format(node, self.num_vars + 4))
                node_continuation_file.write("{} {}\n".format(node, self.num_vars + 5))

            self.node_selected[node] = self.num_vars
            self.node_appear[node] = self.num_vars + 1
            self.node_disappear[node] = self.num_vars + 2
            self.node_split[node] = self.num_vars + 3
            self.node_child[node] = self.num_vars + 4
            self.node_continuation[node] = self.num_vars + 5
            self.num_vars += 6

        if self.write_struct_svm:
            node_selected_file.close()
            node_appear_file.close()
            node_disappear_file.close()
            node_split_file.close()
            node_child_file.close()
            node_continuation_file.close()

        if self.write_struct_svm:
            edge_selected_file = open(f"{self.write_struct_svm}/edge_selected_b{self.block_id}", 'w')
        for edge in self.graph.edges():
            if self.write_struct_svm:
                edge_selected_file.write("{} {} {}\n".format(edge[0], edge[1], self.num_vars))

            self.edge_selected[edge] = self.num_vars
            self.num_vars += 1

        if self.write_struct_svm:
            edge_selected_file.close()

    def _set_objective(self):

        logger.debug("setting objective")

        objective = pylp.LinearObjective(self.num_vars)

        # node selection and cell cycle costs
        if self.write_struct_svm:
            node_selected_weight_file = open(
                f"{self.write_struct_svm}/features_node_selected_weight_b{self.block_id}", 'w')
            node_selected_constant_file = open(
                f"{self.write_struct_svm}/features_node_selected_constant_b{self.block_id}", 'w')
            node_split_weight_file = open(
                f"{self.write_struct_svm}/features_node_split_weight_b{self.block_id}", 'w')
            node_split_constant_file = open(
                f"{self.write_struct_svm}/features_node_split_constant_b{self.block_id}", 'w')
            node_child_weight_or_constant_file = open(
                f"{self.write_struct_svm}/features_node_child_weight_or_constant_b{self.block_id}", 'w')
            node_continuation_weight_or_constant_file = open(
                f"{self.write_struct_svm}/features_node_continuation_weight_or_constant_b{self.block_id}", 'w')
        else:
            node_selected_weight_file = None
            node_selected_constant_file = None
            node_split_weight_file = None
            node_split_constant_file = None
            node_child_weight_or_constant_file = None
            node_continuation_weight_or_constant_file = None

        for node in self.graph.nodes:
            objective.set_coefficient(
                self.node_selected[node],
                self._node_costs(node,
                                 node_selected_weight_file,
                                 node_selected_constant_file))
            objective.set_coefficient(
                self.node_split[node],
                self._split_costs(node,
                                  node_split_weight_file,
                                  node_split_constant_file))
            objective.set_coefficient(
                self.node_child[node],
                self._child_costs(
                    node,
                    node_child_weight_or_constant_file))
            objective.set_coefficient(
                self.node_continuation[node],
                self._continuation_costs(
                    node,
                    node_continuation_weight_or_constant_file))

        if self.write_struct_svm:
            node_selected_weight_file.close()
            node_selected_constant_file.close()
            node_split_weight_file.close()
            node_split_constant_file.close()
            node_child_weight_or_constant_file.close()
            node_continuation_weight_or_constant_file.close()

        # edge selection costs
        if self.write_struct_svm:
            edge_selected_weight_file = open(
                f"{self.write_struct_svm}/features_edge_selected_weight_b{self.block_id}", 'w')
        else:
            edge_selected_weight_file = None

        for edge in self.graph.edges():
            objective.set_coefficient(
                self.edge_selected[edge],
                self._edge_costs(edge,
                                 edge_selected_weight_file))
        if self.write_struct_svm:
            edge_selected_weight_file.close()

        # node appear (skip first frame)
        if self.write_struct_svm:
            appear_file = open(f"{self.write_struct_svm}/features_node_appear_b{self.block_id}", 'w')
            disappear_file = open(f"{self.write_struct_svm}/features_node_disappear_b{self.block_id}", 'w')
        for t in range(self.start_frame + 1, self.end_frame):
            for node in self.graph.cells_by_frame(t):
                objective.set_coefficient(
                    self.node_appear[node],
                    self.parameters.track_cost)
                if self.write_struct_svm:
                    appear_file.write("{} 1\n".format(self.node_appear[node]))
                    disappear_file.write("{} 0\n".format(self.node_disappear[node]))
        for node in self.graph.cells_by_frame(self.start_frame):
            objective.set_coefficient(
                self.node_appear[node],
                0)
            if self.write_struct_svm:
                appear_file.write("{} 0\n".format(self.node_appear[node]))
                disappear_file.write("{} 0\n".format(self.node_disappear[node]))

        # remove node appear costs at edge of roi
        if self.check_node_close_to_roi:
            for node, data in self.graph.nodes(data=True):
                if self._check_node_close_to_roi_edge(
                        node,
                        data,
                        self.parameters.max_cell_move):
                    objective.set_coefficient(
                            self.node_appear[node],
                            0)
                    if self.write_struct_svm:
                        appear_file.write("{} 0\n".format(
                            self.node_appear[node]))

        if self.write_struct_svm:
            appear_file.close()
            disappear_file.close()

        self.objective = objective

    def _check_node_close_to_roi_edge(self, node, data, distance):
        '''Return true if node is within distance to the z,y,x edge
        of the roi. Assumes 4D data with t,z,y,x'''
        if isinstance(distance, dict):
            distance = min(distance.values())

        begin = self.graph.roi.get_begin()[1:]
        end = self.graph.roi.get_end()[1:]
        for index, dim in enumerate(['z', 'y', 'x']):
            node_dim = data[dim]
            begin_dim = begin[index]
            end_dim = end[index]
            if node_dim + distance >= end_dim or\
                    node_dim - distance < begin_dim:
                logger.debug("Node %d with value %s in dimension %s "
                             "is within %s of range [%d, %d]" %
                             (node, node_dim, dim, distance,
                              begin_dim, end_dim))
                return True
        logger.debug("Node %d with position [%s, %s, %s] is not within "
                     "%s to edge of roi %s" %
                     (node, data['z'], data['y'], data['x'], distance,
                         self.graph.roi))
        return False

    def _node_costs(self, node, file_weight, file_constant):
        # node score times a weight plus a threshold
        feature = self.graph.nodes[node]['score']
        if self.feature_func == np.log:
            feature += 0.001
        feature = self.feature_func(feature)
        score_costs = ((feature *
                        self.parameters.weight_node_score) +
                       self.parameters.selection_constant)

        if self.write_struct_svm:
            file_weight.write("{} {}\n".format(
                self.node_selected[node],
                feature))
            file_constant.write("{} 1\n".format(self.node_selected[node]))

        return score_costs

    def _split_costs(self, node, file_weight, file_constant):
        # split score times a weight plus a threshold
        if self.parameters.cell_cycle_key is None:
            if self.write_struct_svm:
                file_constant.write("{} 1\n".format(self.node_split[node]))
                file_weight.write("{} 0\n".format(self.node_split[node]))
            return 1
        feature = self.graph.nodes[node][self.parameters.cell_cycle_key+"mother"]
        if self.feature_func == np.log:
            feature += 0.001
        feature = self.feature_func(feature)
        split_costs = (
            (
                # self.graph.nodes[node][self.parameters.cell_cycle_key][0] *
                feature *
             self.parameters.weight_division) +
            self.parameters.division_constant)

        if self.write_struct_svm:
            file_weight.write("{} {}\n".format(
                self.node_split[node],
                # self.graph.nodes[node][self.parameters.cell_cycle_key][0]
                feature
            ))
            file_constant.write("{} 1\n".format(self.node_split[node]))

        return split_costs

    def _child_costs(self, node, file_weight_or_constant):
        # split score times a weight
        if self.parameters.cell_cycle_key is None:
            if self.write_struct_svm:
                file_weight_or_constant.write("{} 0\n".format(
                    self.node_child[node]))
            return 0
        feature = self.graph.nodes[node][self.parameters.cell_cycle_key+"daughter"]
        if self.feature_func == np.log:
            feature += 0.001
        feature = self.feature_func(feature)
        split_costs = (
            # self.graph.nodes[node][self.parameters.cell_cycle_key][1] *
            feature *
            self.parameters.weight_child)

        if self.write_struct_svm:
            file_weight_or_constant.write("{} {}\n".format(
                self.node_child[node],
                # self.graph.nodes[node][self.parameters.cell_cycle_key][1]
                feature
            ))

        return split_costs

    def _continuation_costs(self, node, file_weight_or_constant):
        # split score times a weight
        if self.parameters.cell_cycle_key is None:
            if self.write_struct_svm:
                file_weight_or_constant.write("{} 0\n".format(
                    self.node_continuation[node]))
            return 0
        feature = self.graph.nodes[node][self.parameters.cell_cycle_key+"normal"]
        if self.feature_func == np.log:
            feature += 0.001
        feature = self.feature_func(feature)
        continuation_costs = (
            # self.graph.nodes[node][self.parameters.cell_cycle_key][2] *
            feature *
            self.parameters.weight_continuation)

        if self.write_struct_svm:
            file_weight_or_constant.write("{} {}\n".format(
                self.node_continuation[node],
                # self.graph.nodes[node][self.parameters.cell_cycle_key][2]
                feature
            ))

        return continuation_costs

    def _edge_costs(self, edge, file_weight):
        # edge score times a weight
        # TODO: normalize node and edge scores to a specific range and
        # ordinality?
        feature = self.graph.edges[edge]['prediction_distance']
        if self.feature_func == np.log:
            feature += 0.001
        feature = self.feature_func(feature)
        edge_costs = (feature *
                      self.parameters.weight_edge_score)

        if self.write_struct_svm:
            file_weight.write("{} {}\n".format(
                self.edge_selected[edge],
                feature))

        return edge_costs

    def _create_constraints(self):

        self.main_constraints = []

        self._add_edge_constraints()
        self._add_cell_cycle_constraints()

        for t in range(self.graph.begin, self.graph.end):
            self._add_inter_frame_constraints(t)

        if self.add_node_density_constraints:
            self._add_node_density_constraints_objective()


    def _add_pin_constraints(self):

        for e in self.graph.edges():

            if self.selected_key in self.graph.edges[e]:

                selected = self.graph.edges[e][self.selected_key]
                self.pinned_edges[e] = selected

                ind_e = self.edge_selected[e]
                constraint = pylp.LinearConstraint()
                constraint.set_coefficient(ind_e, 1)
                constraint.set_relation(pylp.Relation.Equal)
                constraint.set_value(1 if selected else 0)
                self.pin_constraints.append(constraint)

    def _add_edge_constraints(self):

        logger.debug("setting edge constraints")

        if self.write_struct_svm:
            edge_constraint_file = open(f"{self.write_struct_svm}/constraints_edge_b{self.block_id}", 'w')
            cnstr = "2*{} -1*{} -1*{} <= 0\n"
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
            self.main_constraints.append(constraint)

            if self.write_struct_svm:
                edge_constraint_file.write(cnstr.format(ind_e, ind_u, ind_v))

            logger.debug("set edge constraint %s", constraint)

        if self.write_struct_svm:
            edge_constraint_file.close()

    def _add_inter_frame_constraints(self, t):
        '''Linking constraints from t to t+1.'''

        logger.debug("setting inter-frame constraints for frame %d", t)

        # Every selected node has exactly one selected edge to the previous and
        # one or two to the next frame. This includes the special "appear" and
        # "disappear" edges.
        if self.write_struct_svm:
            node_edge_constraint_file = open(f"{self.write_struct_svm}/constraints_node_edge_b{self.block_id}", 'a')
        for node in self.graph.cells_by_frame(t):
            # we model this as three constraints:
            #  sum(prev) -   node  = 0 # exactly one prev edge,
            #                               iff node selected
            #  sum(next) - 2*node <= 0 # at most two next edges
            # -sum(next) +   node <= 0 # at least one next, iff node selected

            constraint_prev = pylp.LinearConstraint()
            constraint_next_1 = pylp.LinearConstraint()
            constraint_next_2 = pylp.LinearConstraint()

            if self.write_struct_svm:
                cnstr_prev = ""
                cnstr_next_1 = ""
                cnstr_next_2 = ""

            # sum(prev)

            # all neighbors in previous frame
            pinned_to_1 = []
            for edge in self.graph.prev_edges(node):
                constraint_prev.set_coefficient(self.edge_selected[edge], 1)
                if self.write_struct_svm:
                    cnstr_prev += "1*{} ".format(self.edge_selected[edge])
                if edge in self.pinned_edges and self.pinned_edges[edge]:
                    pinned_to_1.append(edge)
            if len(pinned_to_1) > 1:
                raise RuntimeError(
                    "Node %d has more than one prev edge pinned: %s"
                    % (node, pinned_to_1))
            # plus "appear"
            constraint_prev.set_coefficient(self.node_appear[node], 1)
            if self.write_struct_svm:
                cnstr_prev += "1*{} ".format(self.node_appear[node])

            # sum(next)

            for edge in self.graph.next_edges(node):
                constraint_next_1.set_coefficient(self.edge_selected[edge], 1)
                constraint_next_2.set_coefficient(self.edge_selected[edge], -1)
                if self.write_struct_svm:
                    cnstr_next_1 += "1*{} ".format(self.edge_selected[edge])
                    cnstr_next_2 += "-1*{} ".format(self.edge_selected[edge])
            # plus "disappear"
            constraint_next_1.set_coefficient(self.node_disappear[node], 1)
            constraint_next_2.set_coefficient(self.node_disappear[node], -1)
            if self.write_struct_svm:
                cnstr_next_1 += "1*{} ".format(self.node_disappear[node])
                cnstr_next_2 += "-1*{} ".format(self.node_disappear[node])
            # node

            constraint_prev.set_coefficient(self.node_selected[node], -1)
            constraint_next_1.set_coefficient(self.node_selected[node], -2)
            constraint_next_2.set_coefficient(self.node_selected[node], 1)
            if self.write_struct_svm:
                cnstr_prev += "-1*{} ".format(self.node_selected[node])
                cnstr_next_1 += "-2*{} ".format(self.node_selected[node])
                cnstr_next_2 += "1*{} ".format(self.node_selected[node])
            # relation, value

            constraint_prev.set_relation(pylp.Relation.Equal)
            constraint_next_1.set_relation(pylp.Relation.LessEqual)
            constraint_next_2.set_relation(pylp.Relation.LessEqual)
            if self.write_struct_svm:
                cnstr_prev += " == "
                cnstr_next_1 += " <= "
                cnstr_next_2 += " <= "

            constraint_prev.set_value(0)
            constraint_next_1.set_value(0)
            constraint_next_2.set_value(0)
            if self.write_struct_svm:
                cnstr_prev += " 0\n"
                cnstr_next_1 += " 0\n"
                cnstr_next_2 += " 0\n"

            self.main_constraints.append(constraint_prev)
            self.main_constraints.append(constraint_next_1)
            self.main_constraints.append(constraint_next_2)

            if self.write_struct_svm:
                node_edge_constraint_file.write(cnstr_prev)
                node_edge_constraint_file.write(cnstr_next_1)
                node_edge_constraint_file.write(cnstr_next_2)

            logger.debug(
                "set inter-frame constraints:\t%s\n\t%s\n\t%s",
                constraint_prev, constraint_next_1, constraint_next_2)

        if self.write_struct_svm:
            node_edge_constraint_file.close()

        # Ensure that the split indicator is set for every cell that splits
        # into two daughter cells.
        if self.write_struct_svm:
            node_split_constraint_file = open(f"{self.write_struct_svm}/constraints_node_split_b{self.block_id}", 'a')
        for node in self.graph.cells_by_frame(t):

            # I.e., each node with two forwards edges is a split node.

            # Constraint 1
            # sum(forward edges) - split   <= 1
            # sum(forward edges) >  1 => split == 1

            # Constraint 2
            # sum(forward edges) - 2*split >= 0
            # sum(forward edges) <= 1 => split == 0

            constraint_1 = pylp.LinearConstraint()
            constraint_2 = pylp.LinearConstraint()
            if self.write_struct_svm:
                cnstr_1 = ""
                cnstr_2 = ""

            # sum(forward edges)
            for edge in self.graph.next_edges(node):
                constraint_1.set_coefficient(self.edge_selected[edge], 1)
                constraint_2.set_coefficient(self.edge_selected[edge], 1)
                if self.write_struct_svm:
                    cnstr_1 += "1*{} ".format(self.edge_selected[edge])
                    cnstr_2 += "1*{} ".format(self.edge_selected[edge])

            # -[2*]split
            constraint_1.set_coefficient(self.node_split[node], -1)
            constraint_2.set_coefficient(self.node_split[node], -2)
            if self.write_struct_svm:
                cnstr_1 += "-1*{} ".format(self.node_split[node])
                cnstr_2 += "-2*{} ".format(self.node_split[node])

            constraint_1.set_relation(pylp.Relation.LessEqual)
            constraint_2.set_relation(pylp.Relation.GreaterEqual)
            if self.write_struct_svm:
                cnstr_1 += " <= "
                cnstr_2 += " >= "

            constraint_1.set_value(1)
            constraint_2.set_value(0)
            if self.write_struct_svm:
                cnstr_1 += " 1\n"
                cnstr_2 += " 0\n"

            self.main_constraints.append(constraint_1)
            self.main_constraints.append(constraint_2)

            if self.write_struct_svm:
                node_split_constraint_file.write(cnstr_1)
                node_split_constraint_file.write(cnstr_2)

            logger.debug(
                "set split-indicator constraints:\n\t%s\n\t%s",
                constraint_1, constraint_2)

        if self.write_struct_svm:
            node_split_constraint_file.close()

    def _add_cell_cycle_constraints(self):
        # If an edge is selected, the division and child indicators are
        # linked. Let e=(u,v) be an edge linking node u at time t + 1 to v in
        # time t.
        # Constraints:
        # child(u) + selected(e) - split(v) <= 1
        # split(v) + selected(e) - child(u) <= 1

        if self.write_struct_svm:
            edge_split_constraint_file = open(f"{self.write_struct_svm}/constraints_edge_split_b{self.block_id}", 'a')
        for e in self.graph.edges():

            # if e is selected, u and v have to be selected
            u, v = e
            ind_e = self.edge_selected[e]
            split_v = self.node_split[v]
            child_u = self.node_child[u]

            link_constraint_1 = pylp.LinearConstraint()
            link_constraint_1.set_coefficient(child_u, 1)
            link_constraint_1.set_coefficient(ind_e, 1)
            link_constraint_1.set_coefficient(split_v, -1)
            link_constraint_1.set_relation(pylp.Relation.LessEqual)
            link_constraint_1.set_value(1)
            self.main_constraints.append(link_constraint_1)
            if self.write_struct_svm:
                link_cnstr_1 = ""
                link_cnstr_1 += "1*{} ".format(child_u)
                link_cnstr_1 += "1*{} ".format(ind_e)
                link_cnstr_1 += "-1*{} ".format(split_v)
                link_cnstr_1 += " <= "
                link_cnstr_1 += " 1\n"
                edge_split_constraint_file.write(link_cnstr_1)

            link_constraint_2 = pylp.LinearConstraint()
            link_constraint_2.set_coefficient(split_v, 1)
            link_constraint_2.set_coefficient(ind_e, 1)
            link_constraint_2.set_coefficient(child_u, -1)
            link_constraint_2.set_relation(pylp.Relation.LessEqual)
            link_constraint_2.set_value(1)
            self.main_constraints.append(link_constraint_2)
            if self.write_struct_svm:
                link_cnstr_2 = ""
                link_cnstr_2 += "1*{} ".format(split_v)
                link_cnstr_2 += "1*{} ".format(ind_e)
                link_cnstr_2 += "-1*{} ".format(child_u)
                link_cnstr_2 += " <= "
                link_cnstr_2 += " 1\n"
                edge_split_constraint_file.write(link_cnstr_2)

        if self.write_struct_svm:
            edge_split_constraint_file.close()

        # Every selected node must be a split, child or continuation
        # (exclusively). If a node is not selected, all the cell cycle
        # indicators should be zero.
        # Constraint for each node:
        # split + child + continuation - selected = 0
        if self.write_struct_svm:
            node_cell_cycle_constraint_file = open(f"{self.write_struct_svm}/constraints_node_cell_cycle_b{self.block_id}", 'a')
        for node in self.graph.nodes():
            cycle_set_constraint = pylp.LinearConstraint()
            cycle_set_constraint.set_coefficient(self.node_split[node], 1)
            cycle_set_constraint.set_coefficient(self.node_child[node], 1)
            cycle_set_constraint.set_coefficient(self.node_continuation[node],
                                                 1)
            cycle_set_constraint.set_coefficient(self.node_selected[node], -1)
            cycle_set_constraint.set_relation(pylp.Relation.Equal)
            cycle_set_constraint.set_value(0)
            self.main_constraints.append(cycle_set_constraint)
            if self.write_struct_svm:
                cc_cnstr = ""
                cc_cnstr += "1*{} ".format(self.node_split[node])
                cc_cnstr += "1*{} ".format(self.node_child[node])
                cc_cnstr += "1*{} ".format(self.node_continuation[node])
                cc_cnstr += "-1*{} ".format(self.node_selected[node])
                cc_cnstr += " == "
                cc_cnstr += " 0\n"
                node_cell_cycle_constraint_file.write(link_cnstr_2)

        if self.write_struct_svm:
            node_cell_cycle_constraint_file.close()

    def _add_node_density_constraints_objective(self):
        logger.debug("adding cell density constraints")
        from scipy.spatial import cKDTree
        import numpy as np
        try:
            nodes_by_t = {
                t: [
                    (
                        node,
                        np.array([data[d] for d in ['z', 'y', 'x']]),
                    )
                    for node, data in self.graph.nodes(data=True)
                    if 't' in data and data['t'] == t
                ]
                for t in range(self.start_frame, self.end_frame)
            }
        except:
            for node, data in self.graph.nodes(data=True):
                print(node, data)
            raise

        rad = 15
        dia = 2*rad
        filter_sz = 1*dia
        r = filter_sz/2
        if isinstance(self.add_node_density_constraints, dict):
            radius = self.add_node_density_constraints
        else:
            radius = {30: 35, 60: 25, 100: 15, 1000:10}
        if self.write_struct_svm:
            node_density_constraint_file = open(f"{self.write_struct_svm}/constraints_node_density_b{self.block_id}", 'w')
        for t in range(self.start_frame, self.end_frame):
            kd_data = [pos for _, pos in nodes_by_t[t]]
            kd_tree = cKDTree(kd_data)

            if isinstance(radius, dict):
                for th in sorted(list(radius.keys())):
                    if t < int(th):
                        r = radius[th]
                        break
            nn_nodes = kd_tree.query_ball_point(kd_data, r,
                                                return_length=False)

            for idx, (node, _) in enumerate(nodes_by_t[t]):
                if len(nn_nodes[idx]) == 1:
                    continue
                constraint = pylp.LinearConstraint()
                if self.write_struct_svm:
                    cnstr = ""
                logger.debug("new constraint (frame %s) node pos %s (node %s)",
                             t, kd_data[idx], node)
                for nn_id in nn_nodes[idx]:
                    if nn_id == idx:
                        continue
                    nn = nodes_by_t[t][nn_id][0]
                    constraint.set_coefficient(self.node_selected[nn], 1)
                    if self.write_struct_svm:
                        cnstr += "1*{} ".format(self.node_selected[nn])
                    logger.debug(
                        "neighbor pos %s %s (node %s)",
                        kd_data[nn_id],
                        np.linalg.norm(np.array(kd_data[idx]) -
                                       np.array(kd_data[nn_id]),
                                       ),
                        nn)
                constraint.set_coefficient(self.node_selected[node], 1)
                constraint.set_relation(pylp.Relation.LessEqual)
                constraint.set_value(1)
                self.main_constraints.append(constraint)
                if self.write_struct_svm:
                    cnstr += "1*{} ".format(self.node_selected[node])
                    cnstr += " <= "
                    cnstr += " 1\n"
                    node_density_constraint_file.write(cnstr)

        if self.write_struct_svm:
            node_density_constraint_file.close()
