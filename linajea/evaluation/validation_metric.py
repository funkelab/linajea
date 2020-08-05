import math
import numpy as np
import networkx as nx
import logging
import time

logger = logging.getLogger(__name__)


def track_distance(track1, track2):
    # Sort nodes by frame (assumed 1 per frame)
    nodes1 = [d for n, d in track1.nodes(data=True)]
    nodes2 = [d for n, d in track2.nodes(data=True)]
    nodes1 = sorted(nodes1, key=lambda n: n['t'])
    nodes2 = sorted(nodes2, key=lambda n: n['t'])

    # Handle empty track edge case
    if len(nodes1) == 0 or len(nodes2) == 0:
        return len(nodes1) + len(nodes2)

    # ensure nodes1 starts earlier in time
    if nodes2[0]['t'] < nodes1[0]['t']:
        tmp = nodes2
        nodes2 = nodes1
        nodes1 = tmp

    dist = 0.
    i1 = 0
    i2 = 0
    while i1 < len(nodes1) and i2 < len(nodes2):
        t1 = nodes1[i1]['t']
        t2 = nodes2[i2]['t']
        if t1 == t2:
            logger.debug("Match in frame %d", t1)
            node_dist = np.linalg.norm(node_loc(nodes1[i1]) -
                                       node_loc(nodes2[i2]))
            logger.debug("Euclidean distance: %f", node_dist)
            logger.debug("normalized distance: %f",
                         norm_distance(node_dist))
            dist += norm_distance(node_dist)
            i1 += 1
            i2 += 1
        else:
            # assumed t1 < t2
            logger.debug("No match in frame %d", t1)
            dist += 1
            i1 += 1

    if i1 < len(nodes1):
        logger.debug("No match from frame %d to %d", i1, len(nodes1) - 1)
        dist += (len(nodes1) - i1)
    if i2 < len(nodes2):
        logger.debug("No match from frame %d to %d", i2, len(nodes2) - 1)
        dist += (len(nodes2) - i2)
    logger.debug("Final distance: %f", dist)
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


def norm_distance(dist, inflect_point=50):
    ''' Normalize the distance to between 0 and 1 using the logistic
    function. The function will be adjusted so that the value at distance zero
    is 10^-6. Due to symmetry, the value at 2*inflect_point will be 1-10^-6.

    Args:
        dist (float)

        inflect_point (float):
            The distance at which the logstic function will be 0.5. Should be
            related to the matching distance penalty - a higher inflect_point
            will penalize the node matching distance less

    '''
    val_at_zero = 0.000001
    slope = math.log(1/val_at_zero - 1) / inflect_point
    logger.debug("Calculated slope: %f" % slope)
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
    conn_components = get_connected_components(lineages)
    logger.info("Number of connected components: %d" % len(conn_components))
    return conn_components


def get_connected_components(graph):
    subgraphs = []
    node_set_generator = nx.weakly_connected_components(graph)
    for node_set in node_set_generator:
        edge_set = graph.edges(node_set, data=True)
        g = nx.DiGraph()
        g.add_nodes_from([(node, graph.nodes[node]) for node in node_set])
        g.add_edges_from(edge_set)
        subgraphs.append(g)
    return subgraphs


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
    processed = 0
    start_time = time.time()
    for gt_track in gt_tracks:
        track_score = None
        for rec_track in rec_tracks:
            s = track_distance(gt_track, rec_track)
            if track_score is None or s < track_score:
                track_score = s
        total_score += track_score
        processed += 1
        if processed % 25 == 0:
            logging.info("Processed %d gt tracks in %d seconds"
                         % (processed, time.time() - start_time))
    return total_score
