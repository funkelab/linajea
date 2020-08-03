import math
import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)


def track_distance(track1, track2):
    start1, end1 = get_node_attr_range(track1, 't')
    start2, end2 = get_node_attr_range(track2, 't')
    if start1 is None:
        return end2 - start2
    if start2 is None:
        return end1 - start1
    if start1 > end2 or start2 > end1:
        # No frame overlap -> length of two tracks
        return (end1 - start1) + (end2 - start2)
    dist = 0.
    if start1 != start2:
        dist += abs(start1 - start2)
        logger.debug("starts don't align. Distance: %f" % dist)
    start_frame = max(start1, start2)
    if end1 != end2:
        dist += abs(end1 - end2)
        logger.debug("ends don't align. Distance: %f" % dist)
    end_frame = min(end1, end2)
    for node1, data1 in track1.nodes(data=True):
        frame = data1['t']
        if frame < start_frame or frame >= end_frame:
            continue
        for node2, data2 in track2.nodes(data=True):
            if data2['t'] == frame:
                logger.debug("Match in frame %d" % frame)
                node_dist = np.linalg.norm(node_loc(data1) -
                                           node_loc(data2))
                logger.debug("Euclidean distance: %f" % node_dist)
                logger.debug("normalized distance: %f" %
                             norm_distance(node_dist))
                dist += norm_distance(node_dist)
                break
    logger.debug("Final distance: %f" % dist)
    return dist


def node_loc(data):
    return np.array([data['z'], data['y'], data['x']])


def get_node_attr_range(graph, attr):
    ''' Returns the lowest value and one greater than the highest value'''
    low = None
    high = None
    for node, data in graph.nodes(data=True):
        value = data[attr]
        if low is None or value < low:
            low = value
        if high is None or value > high:
            high = value
    if low is None:
        return [low, high]
    return [low, high + 1]


def norm_distance(dist, inflect_point=20, slope=1.):
    ''' Normalize the distance to between 0 and 1 using the logistic
    function.

    Args:
        dist (float)

        inflect_point (float):
            The distance at which the logstic function will be 0.5. Should be
            related to the matching distance penalty - a higher inflect_point
            will penalize the node matching distance less

        slope (float):
            Controls the 'slope' or growth rate of the logistic function.
            Lowering this will flatten and stretch out the curve, making the
            distance from inflect_point required to saturate at 0 or 1 larger.
    '''
    return 1. / (1 + math.pow(math.e, -1*slope*(dist - inflect_point)))


def split_into_tracks(lineages):
    ''' Splits a lineage forest into a list of tracks, splitting at divisions
    Args:
        lineages (nx.DiGraph)

    Returns: list(nx.DiGraph)
    '''
    degrees = lineages.in_degree()
    div_nodes = [node for node, degree in degrees if degree == 2]
    logger.debug("Division nodes: %s" % str(div_nodes))
    for node in div_nodes:
        in_edges = list(lineages.in_edges(node))
        min_id = 0
        for edge in in_edges:
            min_id = replace_target(edge, lineages, i=min_id)

    conn_components = [lineages.subgraph(c).copy()
                       for c in nx.weakly_connected_components(lineages)]
    logger.debug("Number of connected components: %d" % len(conn_components))
    return conn_components


def replace_target(edge, graph, i=0):
    old_id = edge[1]
    node_data = graph.nodes[old_id]
    edge_data = graph.edges[edge]
    new_id = get_unused_node_id(graph, i)
    graph.add_node(new_id, **node_data)
    logger.debug("New node has data %s" % graph.nodes[new_id])
    graph.remove_edge(*edge)
    graph.add_edge(edge[0], new_id, **edge_data)
    return new_id


def get_unused_node_id(graph, i=0):
    while i in graph.nodes:
        i += 1
    return i


def validation_score(gt_lineages, rec_lineages):
    ''' Args:

        gt_lineages (networkx.DiGraph)
            Ground truth cell lineages. Assumed to be sparse

        rec_lineages (networkx.DiGraph)
            Reconstructed cell lineages

    Returns:
        A float value that reflects the quality of a set of reconstructed
        lineages. A lower score indicates higher quality. The score suffers a
        high penalty for topological errors (FN edges, FN divisions, FP
        divisions) and a lower penalty for having a large matching distance
        between nodes in the GT and rec tracks.
    '''
    gt_tracks = split_into_tracks(gt_lineages)
    rec_tracks = split_into_tracks(rec_lineages)

    # This is a naive approach where we compare all pairs of tracks.
    # Filtering by frame and/or xyz region would increase efficiency

    total_score = 0
    for gt_track in gt_tracks:
        track_score = None
        for rec_track in rec_tracks:
            s = track_distance(gt_track, rec_track)
            if track_score is None or s < track_score:
                track_score = s
        total_score += track_score
    return total_score
