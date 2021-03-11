from __future__ import absolute_import
import pylp
import logging
import time

import numpy as np
import scipy.sparse
import scipy.spatial

logger = logging.getLogger(__name__)


def match_nodes(track_graph_x, track_graph_y, matching_threshold):
    '''
    Arguments:

        track_graph_x, track_graph_y (``linajea.TrackGraph``):
            Track graphs with the ground truth (x) and predicted (y)

        matching_threshold:
            If the nodes are within matching_threshold
            real world units, then they are allowed to be matched

    Returns a list of nodes in x, a list of nodes in y, a list of node
    matches [(id_x, id_y), ...] referring to indexes in the returned lists,
    and the number of node false positives
    '''
    begin = min(track_graph_x.get_frames()[0], track_graph_x.get_frames()[0])
    end = max(track_graph_x.get_frames()[1], track_graph_x.get_frames()[1]) + 1

    # a dictionary from frame ->
    #       a dictionary of nodes in x ->
    #           list of (neighboring node in y, distance)

    no_match_cost = 2 * matching_threshold + 1
    node_matches = []
    for t in range(begin, end):
        node_costs = {}
        frame_nodes_x = []
        frame_nodes_y = []
        positions_x = []
        positions_y = []

        # get all nodes and their positions in x and y of the current frame

        frame_nodes_x = track_graph_x.cells_by_frame(t)
        positions_x += [
            [track_graph_x.nodes[n]['z'],
             track_graph_x.nodes[n]['y'],
             track_graph_x.nodes[n]['x']]
            for n in frame_nodes_x
        ]

        frame_nodes_y = track_graph_y.cells_by_frame(t)
        positions_y += [
            [track_graph_y.nodes[n]['z'],
             track_graph_y.nodes[n]['y'],
             track_graph_y.nodes[n]['x']]
            for n in frame_nodes_y
        ]

        if len(frame_nodes_x) == 0 or len(frame_nodes_y) == 0:
            continue

        # get all matching edges and distances between x and y of the current
        # frame

        kd_tree_x = scipy.spatial.cKDTree(positions_x)
        kd_tree_y = scipy.spatial.cKDTree(positions_y)

        neighbors_xy = kd_tree_x.query_ball_tree(kd_tree_y, matching_threshold)
        for i, js in enumerate(neighbors_xy):
            node_x = frame_nodes_x[i]
            for j in js:
                node_y = frame_nodes_y[j]
                distance = np.linalg.norm(
                    np.array(positions_x[i]) -
                    np.array(positions_y[j]))
                node_costs[(node_x, node_y)] = distance
        node_matches_in_frame, cost = match(node_costs, no_match_cost)
        logger.info(
                "Done matching frame %d, found %d matches",
                t, len(node_matches_in_frame))
        node_matches.extend(node_matches_in_frame)
    logger.info("Done matching, found %d matches and %d edge fps",
                len(node_matches))
    return node_matches


def match(costs, no_match_cost):
    ''' Arguments:

        costs (``dict`` from ``tuple`` of ids to ``float``):
            A dictionary from a pair of nodes to the cost of matching
            those nodes together. Assumes nodes with no provided
            cost cannot be matched together.

        no_match_cost (``float``):
            The cost of not matching an edge with anything.
    '''

    node_ids_x = set()
    node_ids_y = set()
    for id_x, id_y in costs.keys():
        node_ids_x.add(id_x)
        node_ids_y.add(id_y)
    node_ids_x = sorted(node_ids_x)
    node_ids_y = sorted(node_ids_y)

    no_match_node_x = max(node_ids_x) + 1
    no_match_node_y = max(node_ids_y) + 1
    node_ids_x.append(no_match_node_x)
    node_ids_y.append(no_match_node_y)

    # default cost must be high enough that it will always choose to
    # not match two nodes rather than match them if there
    # is no cost given for the pair
    default_cost = 2*no_match_cost + 1

    n = len(node_ids_x)
    m = len(node_ids_y)
    num_variables = n * m
    objective = pylp.LinearObjective(num_variables)

    for i, id_x in enumerate(node_ids_x):
        for j, id_y in enumerate(node_ids_y):
            key = (id_x, id_y)
            coeff_index = i * m + j
            if key in costs:
                objective.set_coefficient(coeff_index, costs[key])
            elif id_x == no_match_node_x or id_y == no_match_node_y:
                if id_x == no_match_node_x and id_y == no_match_node_y:
                    continue
                objective.set_coefficient(coeff_index, no_match_cost)
            else:
                objective.set_coefficient(coeff_index, default_cost)

    constraints = pylp.LinearConstraints()

    for i in range(n - 1):
        sum_to_one = pylp.LinearConstraint()
        for j in range(m):
            sum_to_one.set_coefficient(i*m+j, 1.0)
        sum_to_one.set_relation(pylp.Relation.Equal)
        sum_to_one.set_value(1.0)
        constraints.add(sum_to_one)

    for j in range(m - 1):
        sum_to_one = pylp.LinearConstraint()
        for i in range(n):
            sum_to_one.set_coefficient(i*m+j, 1.0)
        sum_to_one.set_relation(pylp.Relation.Equal)
        sum_to_one.set_value(1.0)
        constraints.add(sum_to_one)

    solver = pylp.LinearSolver(
            num_variables, pylp.VariableType.Binary,
            preference=pylp.Preference.Gurobi)
    solver.set_objective(objective)
    solver.set_constraints(constraints)
    solver.set_num_threads(1)
    solver.set_timeout(240)

    logger.info("start solving (num vars %d, num constr. %d, num costs %d)",
                num_variables, len(constraints), len(costs))
    start = time.time()
    solution, message = solver.solve()
    end = time.time()
    logger.info("solving took %f seconds", end-start)
    logger.info("solver message: %s", message)
    sol_cost = solution.get_value()
    logger.info("solution cost: %s", sol_cost)

    matches = []
    for i, id_x in enumerate(node_ids_x):
        if id_x == no_match_node_x:
            continue
        for j, id_y in enumerate(node_ids_y):
            if id_y == no_match_node_y:
                continue
            if solution[i*m+j] > 0.5:
                matches.append((id_x, id_y))

    return matches, sol_cost
