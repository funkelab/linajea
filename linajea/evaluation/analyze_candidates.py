import logging

from scipy.spatial import cKDTree, distance

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
