# -*- coding: UTF-8 -*-
import logging
import pylp

logger = logging.getLogger(__name__)


class Solver(object):
    '''
    Class for initializing and solving the ILP problem for
    creating tracks from candidate nodes and edges using pylp
    '''
    def __init__(self, track_graph, parameters, selected_key, frames=None):
        # frames: [start_frame, end_frame] where start_frame is inclusive
        # and end_frame is exclusive. Defaults to track_graph.begin,
        # track_graph.end

        self.graph = track_graph
        self.parameters = parameters
        self.selected_key = selected_key
        self.start_frame = frames[0] if frames else self.graph.begin
        self.end_frame = frames[1] if frames else self.graph.end

        self.node_selected = {}
        self.edge_selected = {}
        self.node_appear = {}
        self.node_disappear = {}
        self.node_split = {}
        self.pinned_edges = {}

        self.num_vars = None
        self.objective = None
        self.main_constraints = []  # list of LinearConstraint objects
        self.pin_constraints = []  # list of LinearConstraint objects
        self.solver = None

        self._create_indicators()
        self._set_objective()
        self._add_constraints()
        self._create_solver()

    def update_objective(self, parameters, selected_key):
        self.parameters = parameters
        self.selected_key = selected_key

        self._set_objective()
        self.solver.set_objective(self.objective)

        self.pinned_edges = []
        self.pin_constraints = []
        self._add_pin_constraints()
        all_constraints = pylp.LinearConstraints()
        for c in self.main_constraints + self.pin_constraints:
            all_constraints.add(c)
        self.solver.set_constraints(all_constraints)

    def _create_solver(self):
        self.solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
        self.solver.initialize(self.num_vars, pylp.VariableType.Binary)
        self.solver.set_objective(self.objective)
        all_constraints = pylp.LinearConstraints()
        for c in self.main_constraints + self.pin_constraints:
            all_constraints.add(c)
        self.solver.set_constraints(all_constraints)
        self.solver.set_num_threads(1)
        self.solver.set_timeout(120)

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
        for node in self.graph.nodes:
            self.node_selected[node] = self.num_vars
            self.node_appear[node] = self.num_vars + 1
            self.node_disappear[node] = self.num_vars + 2
            self.node_split[node] = self.num_vars + 3
            self.num_vars += 4

        for edge in self.graph.edges():
            self.edge_selected[edge] = self.num_vars
            self.num_vars += 1

    def _set_objective(self):

        logger.debug("setting objective")

        objective = pylp.LinearObjective(self.num_vars)

        # node appear (skip first frame)
        for t in range(self.start_frame + 1, self.end_frame):
            for node in self.graph.cells_by_frame(t):
                objective.set_coefficient(
                    self.node_appear[node],
                    self.parameters.cost_appear)
        for node in self.graph.cells_by_frame(self.start_frame):
            objective.set_coefficient(
                self.node_appear[node],
                0)

        # node disappear (skip last frame)
        for t in range(self.start_frame, self.end_frame - 1):
            for node in self.graph.cells_by_frame(t):
                objective.set_coefficient(
                    self.node_disappear[node],
                    self.parameters.cost_disappear)
        for node in self.graph.cells_by_frame(self.end_frame - 1):
            objective.set_coefficient(
                self.node_disappear[node],
                0)

        # remove node appear and disappear costs at edge of roi
        for node, data in self.graph.nodes(data=True):
            if self._check_node_close_to_roi_edge(
                    node,
                    data,
                    self.parameters.max_cell_move):
                objective.set_coefficient(
                        self.node_appear[node],
                        0)
                objective.set_coefficient(
                        self.node_disappear[node],
                        0)

        # node selection and split costs
        for node in self.graph.nodes:
            objective.set_coefficient(
                self.node_selected[node],
                self._node_costs(node))
            objective.set_coefficient(
                self.node_split[node],
                self.parameters.cost_split)

        # edge selection costs
        for edge in self.graph.edges():
            objective.set_coefficient(
                self.edge_selected[edge],
                self._edge_costs(edge))

        self.objective = objective

    def _check_node_close_to_roi_edge(self, node, data, distance):
        '''Return true if node is within distance to the z,y,x edge
        of the roi. Assumes 4D data with t,z,y,x'''
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

    def _node_costs(self, node):

        # simple linear costs based on the score of a node (negative if above
        # threshold_node_score, positive otherwise)

        score_costs = (
            self.parameters.threshold_node_score -
            self.graph.nodes[node]['score'])

        return score_costs*self.parameters.weight_node_score

    def _edge_costs(self, edge):

        # simple linear costs based on the score of an edge (negative if above
        # threshold_edge_score, positive otherwise)
        score_costs = 0

        prediction_distance_costs = (
            (self.graph.edges[edge]['prediction_distance'] -
             self.parameters.threshold_edge_score) *
            self.parameters.weight_prediction_distance_cost)

        return score_costs + prediction_distance_costs

    def _add_constraints(self):

        self.main_constraints = []
        self.pin_constraints = []

        self._add_pin_constraints()
        self._add_edge_constraints()

        for t in range(self.graph.begin, self.graph.end):
            self._add_inter_frame_constraints(t)

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

            logger.debug("set edge constraint %s", constraint)

    def _add_inter_frame_constraints(self, t):
        '''Linking constraints from t to t+1.'''

        logger.debug("setting inter-frame constraints for frame %d", t)

        # Every selected node has exactly one selected edge to the previous and
        # one or two to the next frame. This includes the special "appear" and
        # "disappear" edges.
        for node in self.graph.cells_by_frame(t):

            # we model this as three constraints:
            #  sum(prev) -   node  = 0 # exactly one prev edge,
            #                               iff node selected
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
                    "Node %d has more than one prev edge pinned: %s"
                    % (node, pinned_to_1))
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

            self.main_constraints.append(constraint_prev)
            self.main_constraints.append(constraint_next_1)
            self.main_constraints.append(constraint_next_2)

            logger.debug(
                "set inter-frame constraints:\t%s\n\t%s\n\t%s",
                constraint_prev, constraint_next_1, constraint_next_2)

        # Ensure that the split indicator is set for every cell that splits
        # into two daughter cells.
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

            # sum(forward edges)
            for edge in self.graph.next_edges(node):
                constraint_1.set_coefficient(self.edge_selected[edge], 1)
                constraint_2.set_coefficient(self.edge_selected[edge], 1)

            # -[2*]split
            constraint_1.set_coefficient(self.node_split[node], -1)
            constraint_2.set_coefficient(self.node_split[node], -2)

            constraint_1.set_relation(pylp.Relation.LessEqual)
            constraint_2.set_relation(pylp.Relation.GreaterEqual)

            constraint_1.set_value(1)
            constraint_2.set_value(0)

            self.main_constraints.append(constraint_1)
            self.main_constraints.append(constraint_2)

            logger.debug(
                "set split-indicator constraints:\n\t%s\n\t%s",
                constraint_1, constraint_2)
