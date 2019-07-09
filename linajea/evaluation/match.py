from __future__ import absolute_import
import pylp
import logging
import numpy as np
import scipy.sparse
import scipy.spatial

logger = logging.getLogger(__name__)


def match_edges(track_graph_x, track_graph_y, matching_threshold):
    '''
    Arguments:

        track_graph_x, track_graph_y (``linajea.TrackGraph``):
            Track graphs with the ground truth (x) and predicted (y)

        matching_threshold:
            If the nodes on both ends of an edge are within matching_threshold
            real world units, then they are allowed to be matched

    Returns a list of edges in x, a list of edges in y, a list of edge
    matches [(id_x, id_y), ...] referring to indexes in the returned lists,
    and a list of edge false positives
    '''
    begin = min(track_graph_x.get_frames()[0], track_graph_x.get_frames()[0])
    end = max(track_graph_x.get_frames()[1], track_graph_x.get_frames()[1]) + 1

    edges_x = [(int(u), int(v)) for u, v in track_graph_x.edges()]
    edges_y = [(int(u), int(v)) for u, v in track_graph_y.edges()]

    # a dictionary from nodes in x to a list of
    # (neighboring node in y, distance)
    node_pairs_xy = {}

    for t in range(begin, end):

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

        kd_tree_x = scipy.spatial.KDTree(positions_x)
        kd_tree_y = scipy.spatial.KDTree(positions_y)

        neighbors_xy = kd_tree_x.query_ball_tree(kd_tree_y, matching_threshold)
        for i, js in enumerate(neighbors_xy):
            node_x = frame_nodes_x[i]
            node_x_neighbors = []
            for j in js:
                node_y = frame_nodes_y[j]
                distance = np.linalg.norm(
                    np.array(positions_x[i]) -
                    np.array(positions_y[j]))
                node_x_neighbors.append((node_y, distance))
            node_pairs_xy[node_x] = node_x_neighbors

    edge_costs = get_edge_costs(edges_x, edges_y, node_pairs_xy)
    y_edges_in_range = set(edge[1] for edge in edge_costs.keys())
    edge_matches, _ = match(edge_costs, 2*matching_threshold + 1)
    edge_fps = len(y_edges_in_range) - len(edge_matches)
    return edges_x, edges_y, edge_matches, edge_fps


def get_edge_costs(edges_x, edges_y, node_pairs_xy):
    '''
    Arguments:
        edges_x, edges_y:
            lists of edges in x and y, where each edge is just (u, v).
            Edge ids in edge_costs will be the index in the list of the edge

        node_pairs_xy:
            a dictionary from nodes in x to a list of
            (neighboring node in y, distance)
    '''

    edges_y_by_source = {}
    for edge_id_y, (u, v) in enumerate(edges_y):
        assert u not in edges_y_by_source,\
                "Each edge should have a unique source node"
        edges_y_by_source[u] = (v, edge_id_y)

    edge_costs = {}
    for edge_id_x, (ux, vx) in enumerate(edges_x):
        if ux not in node_pairs_xy:
            continue
        uys = node_pairs_xy[ux]
        if vx not in node_pairs_xy:
            continue
        vys_and_distances = node_pairs_xy[vx]
        vys = [e[0] for e in vys_and_distances]

        for uy, u_distance in uys:
            if uy not in edges_y_by_source:
                continue
            v, edge_id_y = edges_y_by_source[uy]
            try:
                i = vys.index(v)
            except ValueError:
                continue
            vy, v_distance = vys_and_distances[i]
            edge_costs[(edge_id_x, edge_id_y)] = u_distance + v_distance

    return edge_costs


def match(costs, no_match_cost):
    ''' Arguments:

        costs (``dict`` from ``tuple`` of ids to ``float``):
            A dictionary from a pair of edges to the cost of matching
            those edges together. Assumes edges with no provided
            cost cannot be matched together.

        no_match_cost (``float``):
            The cost of not matching an edge with anything.
    '''

    edge_ids_x = set()
    edge_ids_y = set()
    for id_x, id_y in costs.keys():
        edge_ids_x.add(id_x)
        edge_ids_y.add(id_y)
    edge_ids_x = sorted(edge_ids_x)
    edge_ids_y = sorted(edge_ids_y)

    no_match_edge_x = max(edge_ids_x) + 1
    no_match_edge_y = max(edge_ids_y) + 1
    edge_ids_x.append(no_match_edge_x)
    edge_ids_y.append(no_match_edge_y)

    # default cost must be high enough that it will always choose to
    # not match two edges rather than match them if there
    # is no cost given for the pair
    default_cost = 2*no_match_cost + 1

    n = len(edge_ids_x)
    m = len(edge_ids_y)
    num_variables = n * m
    objective = pylp.LinearObjective(num_variables)

    for i, id_x in enumerate(edge_ids_x):
        for j, id_y in enumerate(edge_ids_y):
            key = (id_x, id_y)
            coeff_index = i * m + j
            if key in costs:
                objective.set_coefficient(coeff_index, costs[key])
            elif id_x == no_match_edge_x or id_y == no_match_edge_y:
                if id_x == no_match_edge_x and id_y == no_match_edge_y:
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

    solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
    solver.initialize(num_variables, pylp.VariableType.Binary)
    solver.set_objective(objective)
    solver.set_constraints(constraints)
    solver.set_num_threads(1)
    solver.set_timeout(240)

    solution, message = solver.solve()
    sol_cost = solution.get_value()

    matches = []
    for i, id_x in enumerate(edge_ids_x):
        if id_x == no_match_edge_x:
            continue
        for j, id_y in enumerate(edge_ids_y):
            if id_y == no_match_edge_y:
                continue
            if solution[i*m+j] > 0.5:
                matches.append((id_x, id_y))

    return matches, sol_cost
