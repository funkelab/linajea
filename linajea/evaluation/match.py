"""Provides function to match edges in two graphs to each other
"""
import logging
import time

import numpy as np
import pylp
import scipy.sparse
import scipy.spatial

logger = logging.getLogger(__name__)


def match_edges(track_graph_x, track_graph_y, matching_threshold):
    '''Perform matching of two graphs based on edges

    Args
    ----

        track_graph_x, track_graph_y (``linajea.TrackGraph``):
            Track graphs with the ground truth (x) and predicted (y)

        matching_threshold:
            If the nodes on both ends of an edge are within matching_threshold
            real world units, then they are allowed to be matched

    Returns
    -------
    list
        A list of edges in x, a list of edges in y, a list of edge
        matches [(id_x, id_y), ...] referring to indexes in the returned
        lists, and the number of edge false positives
    '''
    begin = min(track_graph_x.get_frames()[0], track_graph_x.get_frames()[0])
    end = max(track_graph_x.get_frames()[1], track_graph_x.get_frames()[1]) + 1

    edges_x = [(int(u), int(v)) for u, v in track_graph_x.edges()]
    edges_y = [(int(u), int(v)) for u, v in track_graph_y.edges()]
    edges_y_by_source = {}
    for edge_id_y, (u, v) in enumerate(edges_y):
        assert u not in edges_y_by_source,\
                "Each edge should have a unique source node"
        edges_y_by_source[u] = (v, edge_id_y)

    # a dictionary from frame ->
    #       a dictionary of nodes in x ->
    #           list of (neighboring node in y, distance)
    node_pairs_xy_by_frame = {}
    edge_matches = []
    edge_fps = []

    avg_dist = []
    avg_dist_target = []
    avg_dist_source = []
    for t in range(begin, end):
        node_pairs_xy = {}
        frame_nodes_x = []
        frame_nodes_y = []
        positions_x = []
        positions_y = []

        avg_dist_frame = []
        avg_dist_target_frame = []
        avg_dist_source_frame = []

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
            node_x_neighbors = []
            for j in js:
                node_y = frame_nodes_y[j]
                distance = np.linalg.norm(
                    np.array(positions_x[i]) -
                    np.array(positions_y[j]))
                node_x_neighbors.append((node_y, distance))
            node_pairs_xy[node_x] = node_x_neighbors
        node_pairs_xy_by_frame[t] = node_pairs_xy

        if t - 1 in node_pairs_xy_by_frame.keys():
            logger.debug("finding matches in frame %d" % t)
            node_pairs_two_frames = node_pairs_xy.copy()
            node_pairs_two_frames.update(node_pairs_xy_by_frame[t-1])
            edge_costs = get_edge_costs(
                    edges_x, edges_y_by_source, node_pairs_two_frames)
            if edge_costs == {}:
                logger.info("No potential matches with source in frame %d" % t)
                continue
            logger.debug("costs: %s" % edge_costs)
            y_edges_in_range = set(edge[1] for edge in edge_costs.keys())
            logger.debug("Y edges in range: %s" % y_edges_in_range)
            edge_matches_in_frame, _ = match(edge_costs,
                                             2*matching_threshold + 1)
            edge_matches.extend(edge_matches_in_frame)
            y_edge_matches_in_frame = [edge[1]
                                       for edge in edge_matches_in_frame]
            edge_fps_in_frame = set(y_edges_in_range) -\
                set(y_edge_matches_in_frame)
            edge_fps += list(edge_fps_in_frame)
            logger.debug(
                    "Done matching frame %d, found %d matches and %d edge fps",
                    t, len(edge_matches_in_frame), len(edge_fps_in_frame))

            for exid, eyid in edge_matches_in_frame:
                node_xid_source = edges_x[exid][0]
                node_xid_target = edges_x[exid][1]
                node_yid_source = edges_y[eyid][0]
                node_yid_target = edges_y[eyid][1]

                pos_x_target = np.array(
                    [track_graph_x.nodes[node_xid_target]['z'],
                     track_graph_x.nodes[node_xid_target]['y'],
                     track_graph_x.nodes[node_xid_target]['x']])
                pos_y_target = np.array(
                    [track_graph_y.nodes[node_yid_target]['z'],
                     track_graph_y.nodes[node_yid_target]['y'],
                     track_graph_y.nodes[node_yid_target]['x']])
                distance_target = np.linalg.norm(pos_x_target - pos_y_target)
                pos_x_source = np.array(
                    [track_graph_x.nodes[node_xid_source]['z'],
                     track_graph_x.nodes[node_xid_source]['y'],
                     track_graph_x.nodes[node_xid_source]['x']])
                pos_y_source = np.array(
                    [track_graph_y.nodes[node_yid_source]['z'],
                     track_graph_y.nodes[node_yid_source]['y'],
                     track_graph_y.nodes[node_yid_source]['x']])
                distance_source = np.linalg.norm(pos_x_source - pos_y_source)

                avg_dist_source_frame.append(distance_source)
                avg_dist_source.append(distance_source)
                avg_dist_target_frame.append(distance_target)
                avg_dist_target.append(distance_target)
                avg_dist_frame.append(distance_target)
                avg_dist_frame.append(distance_source)
                avg_dist.append(distance_target)
                avg_dist.append(distance_source)
                if distance_target >= 7.0:
                    logger.debug("target %d %d %.3f %s %s", node_xid_target,
                                 node_yid_target, distance_target,
                                 pos_x_target, pos_y_target)
                if distance_source >= 7.0:
                    logger.debug("source %d %d %.3f %s %s", node_xid_source,
                                 node_yid_source, distance_source,
                                 pos_x_source, pos_y_source)
                # logger.info("%.3f %.3f", distance_target, distance_source)

        logger.debug("frame %d, count matches %d",
                     t, len(avg_dist_source_frame))
        if len(avg_dist_source_frame) == 0:
            continue
        logger.debug("dist source: avg %.3f, med %.3f, min %.3f, max %.3f",
                     np.mean(avg_dist_source_frame),
                     np.median(avg_dist_source_frame),
                     np.min(avg_dist_source_frame),
                     np.max(avg_dist_source_frame))
        logger.debug("dist target: avg %.3f, med %.3f, min %.3f, max %.3f",
                     np.mean(avg_dist_target_frame),
                     np.median(avg_dist_target_frame),
                     np.min(avg_dist_target_frame),
                     np.max(avg_dist_target_frame))
        logger.debug("dist : avg %.3f, med %.3f, min %.3f, max %.3f",
                     np.mean(avg_dist_frame),
                     np.median(avg_dist_frame),
                     np.min(avg_dist_frame),
                     np.max(avg_dist_frame))

    if len(avg_dist_source) > 0:
        logger.debug("total count matches %d", len(avg_dist_source))
        logger.debug("dist source: avg %.3f, med %.3f, min %.3f, max %.3f",
                     np.mean(avg_dist_source),
                     np.median(avg_dist_source),
                     np.min(avg_dist_source),
                     np.max(avg_dist_source))
        logger.debug("dist target: avg %.3f, med %.3f, min %.3f, max %.3f",
                     np.mean(avg_dist_target),
                     np.median(avg_dist_target),
                     np.min(avg_dist_target),
                     np.max(avg_dist_target))
        logger.debug("dist : avg %.3f, med %.3f, min %.3f, max %.3f",
                     np.mean(avg_dist),
                     np.median(avg_dist),
                     np.min(avg_dist),
                     np.max(avg_dist))
    logger.info("Done matching, found %d matches and %d edge fps"
                % (len(edge_matches), len(edge_fps)))
    return edges_x, edges_y, edge_matches, edge_fps


def get_edge_costs(edges_x, edges_y_by_source, node_pairs_xy):
    '''
    Arguments:
        edges_x (list of int):
            list of edges in x, where each edge is just (u, v).
            Edge ids in edge_costs will be the list index of each edge

        edges_y_by_source (dict: int -> (int, int)):
            Dictionary of y edges, where each entry is
            u: (v, index from original edges_y list)

        node_pairs_xy (dict: int -> (int, float)):
            A dictionary from nodes in x to a list of
            (neighboring node in y, distance)
    '''

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

    solver = pylp.LinearSolver(
            num_variables,
            pylp.VariableType.Binary,
            preference=pylp.Preference.Any)
    solver.set_objective(objective)
    solver.set_constraints(constraints)
    solver.set_num_threads(1)
    solver.set_timeout(240)

    logger.debug("start solving (num vars %d, num constr. %d, num costs %d)",
                 num_variables, len(constraints), len(costs))
    start = time.time()
    solution, message = solver.solve()
    end = time.time()
    logger.debug("solving took %f seconds", end-start)
    logger.debug("solver message: %s", message)
    sol_cost = solution.get_value()
    logger.debug("solution cost: %s", sol_cost)

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
