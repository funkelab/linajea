# -*- coding: utf-8 -*-
import logging
import pylp

logger = logging.getLogger(__name__)


class NMSolver(object):
    '''
    Class for initializing and solving the ILP problem for
    creating tracks from candidate nodes and edges using pylp.
    This is the "non-minimal" (NM) version, or the original formulation before
    we minimized the number of variables using assumptions about their
    relationships
    '''
    def __init__(self, track_graph, parameters, selected_key, frames=None,
                 check_node_close_to_roi=True, timeout=120,
                 add_node_density_constraints=False):
        # frames: [start_frame, end_frame] where start_frame is inclusive
        # and end_frame is exclusive. Defaults to track_graph.begin,
        # track_graph.end
        self.check_node_close_to_roi = check_node_close_to_roi
        self.add_node_density_constraints = add_node_density_constraints

        self.graph = track_graph
        self.parameters = parameters
        self.selected_key = selected_key
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

        if self.parameters.use_cell_state:
            self.edge_split = {}

        self.num_vars = None
        self.objective = None
        self.main_constraints = []  # list of LinearConstraint objects
        self.pin_constraints = []  # list of LinearConstraint objects
        self.solver = None

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
        for node in self.graph.nodes:
            self.node_selected[node] = self.num_vars
            self.node_appear[node] = self.num_vars + 1
            self.node_disappear[node] = self.num_vars + 2
            self.node_split[node] = self.num_vars + 3
            self.node_child[node] = self.num_vars + 4
            self.node_continuation[node] = self.num_vars + 5
            self.num_vars += 6

        for edge in self.graph.edges():
            self.edge_selected[edge] = self.num_vars
            self.num_vars += 1

            if self.parameters.use_cell_state:
                self.edge_split[edge] = self.num_vars
                self.num_vars += 1

    def _set_objective(self):

        logger.debug("setting objective")

        objective = pylp.LinearObjective(self.num_vars)

        # node selection and split costs
        for node in self.graph.nodes:
            objective.set_coefficient(
                self.node_selected[node],
                self._node_costs(node))
            objective.set_coefficient(
                self.node_split[node],
                self._split_costs(node))
            objective.set_coefficient(
                self.node_child[node],
                self._child_costs(node))
            objective.set_coefficient(
                self.node_continuation[node],
                self._continuation_costs(node))

        # edge selection costs
        for edge in self.graph.edges():
            objective.set_coefficient(
                self.edge_selected[edge],
                self._edge_costs(edge))

            if self.parameters.use_cell_state:
                objective.set_coefficient(
                    self.edge_split[edge], 0)

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
        if self.check_node_close_to_roi:
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

    def _node_costs(self, node):
        # node score times a weight plus a threshold
        score_costs = ((self.parameters.threshold_node_score -
                        self.graph.nodes[node]['score']) *
                       self.parameters.weight_node_score)

        return score_costs

    def _split_costs(self, node):
        if not self.parameters.use_cell_state:
            return self.parameters.cost_split
        elif self.parameters.use_cell_state == 'simple' or \
           self.parameters.use_cell_state == 'v1' or \
           self.parameters.use_cell_state == 'v2':
            return ((self.parameters.threshold_split_score -
                     self.graph.nodes[node][self.parameters.prefix+'mother']) *
                    self.parameters.cost_split)
        elif self.parameters.use_cell_state == 'v3' or \
             self.parameters.use_cell_state == 'v4':
            if self.graph.nodes[node][self.parameters.prefix+'mother'] > \
               self.parameters.threshold_split_score:
                return -self.parameters.cost_split
            else:
                return self.parameters.cost_split
        else:
            raise NotImplementedError("invalid value for use_cell_state")

    def _child_costs(self, node):
        if not self.parameters.use_cell_state:
            return 0
        elif self.parameters.use_cell_state == 'v1' or \
           self.parameters.use_cell_state == 'v2':
            return ((self.parameters.threshold_split_score -
                     self.graph.nodes[node][self.parameters.prefix+'daughter']) *
                    self.parameters.cost_daughter)
        elif self.parameters.use_cell_state == 'v3' or \
             self.parameters.use_cell_state == 'v4':
            if self.graph.nodes[node][self.parameters.prefix+'daughter'] > \
               self.parameters.threshold_split_score:
                return -self.parameters.cost_daughter
            else:
                return self.parameters.cost_daughter
        else:
            raise NotImplementedError("invalid value for use_cell_state")

    def _continuation_costs(self, node):
        if not self.parameters.use_cell_state or \
           self.parameters.use_cell_state == 'v1' or \
           self.parameters.use_cell_state == 'v3':
            return 0
        elif self.parameters.use_cell_state == 'v2':
            return ((self.parameters.threshold_is_normal_score -
                     self.graph.nodes[node][self.parameters.prefix+'normal']) *
                    self.parameters.cost_normal)
        elif self.parameters.use_cell_state == 'v4':
            if self.graph.nodes[node][self.parameters.prefix+'normal'] > \
               self.parameters.threshold_is_normal_score:
                # return 0
                return -self.parameters.cost_normal
            else:
                return self.parameters.cost_normal
        else:
            raise NotImplementedError("invalid value for use_cell_state")

    def _edge_costs(self, edge):

        # simple linear costs based on the score of an edge (negative if above
        # threshold_edge_score, positive otherwise)
        score_costs = 0

        prediction_distance_costs = (
            (self.graph.edges[edge]['prediction_distance'] -
             self.parameters.threshold_edge_score) *
            self.parameters.weight_prediction_distance_cost)

        return score_costs + prediction_distance_costs

    def _create_constraints(self):

        self.main_constraints = []

        self._add_edge_constraints()
        for t in range(self.graph.begin, self.graph.end):
            self._add_cell_cycle_constraints(t)

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

    def _add_cell_cycle_constraints(self, t):
        for node in self.graph.cells_by_frame(t):
            if self.parameters.use_cell_state:
                #  sum(next(edges_split))- 2*split >= 0
                constraint_3 = pylp.LinearConstraint()
                for edge in self.graph.next_edges(node):
                    constraint_3.set_coefficient(self.edge_split[edge], 1)
                constraint_3.set_coefficient(self.node_split[node], -2)
                constraint_3.set_relation(pylp.Relation.Equal)
                constraint_3.set_value(0)

                self.main_constraints.append(constraint_3)

                constraint_4 = pylp.LinearConstraint()
                for edge in self.graph.prev_edges(node):
                    constraint_4.set_coefficient(self.edge_split[edge], 1)
                constraint_4.set_coefficient(self.node_child[node], -1)
                constraint_4.set_relation(pylp.Relation.Equal)
                constraint_4.set_value(0)

                self.main_constraints.append(constraint_4)

            if self.parameters.use_cell_state == 'v2' or \
               self.parameters.use_cell_state == 'v4':
                constraint_6 = pylp.LinearConstraint()
                constraint_6.set_coefficient(self.node_selected[node], -1)
                constraint_6.set_coefficient(self.node_split[node], 1)
                constraint_6.set_coefficient(self.node_child[node], 1)
                constraint_6.set_coefficient(self.node_continuation[node], 1)
                constraint_6.set_relation(pylp.Relation.Equal)
                constraint_6.set_value(0)
                self.main_constraints.append(constraint_6)

    def _add_node_density_constraints_objective(self):
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
        for t in range(self.start_frame, self.end_frame):
            kd_data = [pos for _, pos in nodes_by_t[t]]
            kd_tree = cKDTree(kd_data)

            if isinstance(radius, dict):
                for th in sorted(list(radius.keys())):
                    if t < int(th):
                        r = radius[th]
                        break
            nn_nodes = kd_tree.query_ball_point(kd_data, r, p=np.inf,
                                                return_length=False)

            for idx, (node, _) in enumerate(nodes_by_t[t]):
                if len(nn_nodes[idx]) == 1:
                    continue
                constraint = pylp.LinearConstraint()
                logger.debug("new constraint (frame %s) node pos %s",
                             t, kd_data[idx])
                for nn_id in nn_nodes[idx]:
                    if nn_id == idx:
                        continue
                    nn = nodes_by_t[t][nn_id][0]
                    constraint.set_coefficient(self.node_selected[nn], 1)
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
