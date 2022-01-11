from scipy.spatial import cKDTree, distance
import numpy as np
import logging

logger = logging.getLogger(__name__)
# TODO: Don't use candidate_graphs, networkx is slow and unnecessary for the
# candidates. Go straight to pymongo queries, or maybe daisy mongo node/edge
# queries


def get_kd_trees_by_frame(candidate_db, node_score_threshold=None):

    nodes_by_frame = {}
    for node_id, node in candidate_db.nodes(data=True):
        if node_score_threshold and node['score'] < node_score_threshold:
            continue
        if 't' not in node:
            logger.debug("(%d, %s) has no time, skipping" % (node_id, node))
            continue
        t = node['t']
        if t not in nodes_by_frame:
            nodes_by_frame[t] = []
        nodes_by_frame[t].append([node['z'], node['y'], node['x']])

    kd_trees_by_frame = {}
    for frame, nodes in nodes_by_frame.items():
        kd_trees_by_frame[frame] = cKDTree(nodes)
    return kd_trees_by_frame


def get_node_recall(
        candidate_graph,
        gt_graph,
        match_distance,
        score_threshold=None):
    gt_kd_trees = get_kd_trees_by_frame(gt_graph)
    cand_kd_trees = get_kd_trees_by_frame(candidate_graph, score_threshold)
    num_matches = 0
    num_gt_nodes = len(gt_graph)
    for frame, gt_kd_tree in gt_kd_trees.items():
        if frame not in cand_kd_trees:
            logger.warn("Frame %d not in candidate graph but in gt graph"
                        % frame)
            continue
        cand_kd_tree = cand_kd_trees[frame]
        neighbors = gt_kd_tree.query_ball_tree(cand_kd_tree, match_distance)
        for index, matches in enumerate(neighbors):
            if len(matches) > 0:
                # there is at least one candidate within radius
                num_matches += 1

    return num_matches, num_gt_nodes


def get_edge_recall(
        candidate_graph,
        gt_graph,
        match_distance,
        move_distance,
        node_score_threshold=None,
        edge_score_threshold=None):
    cand_kd_trees = get_kd_trees_by_frame(
            candidate_graph, node_score_threshold)
    num_matches = 0
    num_gt_edges = 0
    for source_id, target_id in gt_graph.edges():
        source_node = gt_graph.nodes[source_id]
        target_node = gt_graph.nodes[target_id]
        if 't' not in target_node:
            logger.warn("Target node %s is not in roi" % target_id)
            continue
        num_gt_edges += 1
        source_frame = source_node['t']
        target_frame = target_node['t']
        assert source_frame - 1 == target_frame
        if source_frame not in cand_kd_trees:
            logger.warn("Frame %s not in candidate graph" % source_frame)
            break
        source_kd_tree = cand_kd_trees[source_frame]
        if target_frame not in cand_kd_trees:
            logger.warn("Frame %s not in candidate graph" % target_frame)
            break
        target_kd_tree = cand_kd_trees[target_frame]
        source_neighbors = source_kd_tree.query_ball_point(
                [source_node[dim] for dim in ['z', 'y', 'x']], match_distance)
        target_neighbors = target_kd_tree.query_ball_point(
                [source_node[dim] for dim in ['z', 'y', 'x']], match_distance)
        matched = False
        for source_match in source_neighbors:
            if matched:
                break
            for target_match in target_neighbors:
                edge_distance = distance.euclidean(
                        source_kd_tree.data[source_match],
                        target_kd_tree.data[target_match])
                if edge_distance < move_distance:
                    matched = True
                    continue
        if matched:
            num_matches += 1
    return num_matches, num_gt_edges


def sort_nodes_by_frame(graph):
    sorted_nodes = {}
    for node_id, node in graph.nodes(data=True):
        if 't' not in node:
            continue
        if node['t'] not in sorted_nodes:
            sorted_nodes[node['t']] = []
        sorted_nodes[node['t']].append((node_id, node))
    return sorted_nodes


def calc_pv_distances(
        candidate_graph,
        gt_graph,
        match_distance):
    gt_nodes_by_frame = sort_nodes_by_frame(gt_graph)
    cand_nodes_by_frame = sort_nodes_by_frame(candidate_graph)
    cand_kd_trees = get_kd_trees_by_frame(candidate_graph)
    prediction_distances = []
    baseline_distances = []
    for frame, cand_kd_tree in cand_kd_trees.items():
        if frame not in gt_nodes_by_frame:
            logger.warn("Frame %s has cand nodes but no gt nodes" % frame)
            continue
        gt_nodes = gt_nodes_by_frame[frame]
        candidate_nodes = cand_nodes_by_frame[frame]
        for gt_node_id, gt_node_info in gt_nodes:
            # get location of real parent
            parent_edges = list(gt_graph.prev_edges(gt_node_id))
            if len(parent_edges) == 0:
                continue
            assert len(parent_edges) == 1
            parent_id = parent_edges[0][1]
            parent_node = gt_graph.nodes[parent_id]
            parent_location = np.array([
                parent_node['z'],
                parent_node['y'],
                parent_node['x']])

            # get predicted location of parent
            gt_node_location = np.array([
                    gt_node_info['z'],
                    gt_node_info['y'],
                    gt_node_info['x']])
            distance, index = cand_kd_tree.query(
                    gt_node_location, k=1, distance_upper_bound=match_distance)
            if distance > match_distance:
                continue
            matched_cand_pv = np.array(candidate_nodes[index]['parent_vector'])
            matched_cand_location = np.array(cand_kd_tree.data[index])
            predicted_location = matched_cand_location + matched_cand_pv
            prediction_distances.append(np.linalg.norm(
                predicted_location - parent_location))

            baseline_distances.append(np.linalg.norm(
                matched_cand_location - parent_location))

        return prediction_distances, baseline_distances
