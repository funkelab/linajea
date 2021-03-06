from __future__ import absolute_import
from .solver import Solver
from .track_graph import TrackGraph
import logging
import time
import networkx as nx

logger = logging.getLogger(__name__)


class TrackingParameters(object):

    def __init__(
            self,
            block_size=None,
            context=None,
            cost_appear=None,
            cost_disappear=None,
            cost_split=None,
            max_cell_move=None,
            threshold_node_score=None,
            weight_node_score=None,
            threshold_edge_score=None,
            weight_prediction_distance_cost=None,
            version=None,
            **kwargs):

        # block size and context
        assert block_size is not None, "Failed to specify block_size"
        self.block_size = block_size
        assert context is not None, "Failed to specify context"
        self.context = context

        # track costs:
        assert cost_appear is not None, "Failed to specify cost_appear"
        self.cost_appear = cost_appear
        assert cost_disappear is not None, "Failed to specify cost_disappear"
        self.cost_disappear = cost_disappear
        assert cost_split is not None, "Failed to specify cost_split"
        self.cost_split = cost_split

        # max_cell_move
        # nodes within this distance to the block boundary will not pay
        # the appear and disappear costs
        # (Should be < 1/2 the context in z/x/y)
        assert max_cell_move is not None, "Failed to specify max_cell_move"
        self.max_cell_move = max_cell_move

        # node costs:

        # nodes with scores below this threshold will have a positive cost,
        # above this threshold a negative cost
        assert threshold_node_score is not None,\
            "Failed to specify threshold_node_score"
        self.threshold_node_score = threshold_node_score

        # scaling factor after the conversion to costs above
        assert weight_node_score is not None,\
            "Failed to specify weight_node_score"
        self.weight_node_score = weight_node_score

        # edge costs:

        # similar to node costs, determines when a cost is positive/negative
        assert threshold_edge_score is not None,\
            "Failed to specify threshold_edge_score"
        self.threshold_edge_score = threshold_edge_score

        # how to weigh the Euclidean distance between the predicted position
        # and the actual position of cells for the costs of an edge
        assert weight_prediction_distance_cost is not None,\
            "Failed to specify weight_prediction_distance_cost"
        self.weight_prediction_distance_cost = weight_prediction_distance_cost
        # version control
        self.version = version


def track(graph, parameters, selected_key, frame_key='t', frames=None):
    ''' A wrapper function that takes a daisy subgraph and input parameters,
    creates and solves the ILP to create tracks, and updates the daisy subgraph
    to reflect the selected nodes and edges.

    Args:

        graph (``daisy.SharedSubgraph``):

            The candidate graph to extract tracks from

        parameters (``TrackingParameters``)

            The parameters to use when optimizing the tracking ILP

        selected_key (``string``)

            The key used to store the `true` or `false` selection status of
            each node and edge in graph.

        frame_key (``string``, optional):

            The name of the node attribute that corresponds to the frame of the
            node. Defaults to "t".

        frames (``list`` of ``int``):

            The start and end frames to solve in (in case the graph doesn't
            have nodes in all frames). Start is inclusive, end is exclusive.
            Defaults to graph.begin, graph.end

    '''
    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        return

    logger.info("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)

    logger.info("Creating solver...")
    solver = Solver(track_graph, parameters, selected_key, frames=frames)

    logger.info("Solving...")
    start_time = time.time()
    solver.solve()
    logger.info("Solving ILP took %s seconds" % (time.time() - start_time))

    for cell, data in graph.nodes(data=True):
        data[selected_key] = track_graph.nodes[cell][selected_key]
    for u, v, data in graph.edges(data=True):
        if (u, v) in track_graph.edges:
            data[selected_key] = track_graph.edges[(u, v)][selected_key]


def greedy_track(
        graph,
        selected_key,
        metric='prediction_distance',
        frame_key='t',
        frames=None,
        node_threshold=None):
    ''' A wrapper function that takes a daisy subgraph and input parameters,
    greedily chooses edges to create tracks, and updates the daisy subgraph to
    reflect the selected nodes and edges.

    Args:

        graph (``daisy.SharedSubgraph``):
            The candidate graph to extract tracks from

        selected_key (``string``)
            The key used to store the `true` or `false` selection status of
            each edge in graph.

        metric (``string``)
            Type of distance to use when finding "shortest" edges. Options are
            'prediction_distance' (default) and 'distance'

        frame_key (``string``, optional):

            The name of the node attribute that corresponds to the frame of the
            node. Defaults to "t".

        frames (``list`` of ``int``):
            The start and end frames to solve in (in case the graph doesn't
            have nodes in all frames). Start is inclusive, end is exclusive.
            Defaults to graph.begin, graph.end

        node_threshold (``float``):
            Don't use nodes with score below this values. Defaults to None.
    '''
    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        return

    selected = nx.DiGraph()
    unselected = nx.DiGraph()
    unselected.add_nodes_from(graph.nodes(data=True))
    unselected.add_edges_from(graph.edges(data=True))
    nx.set_edge_attributes(graph, False, selected_key)

    if node_threshold:
        logger.info("Removing nodes below threshold")
        for node, data in list(unselected.nodes(data=True)):
            if data['score'] < node_threshold:
                unselected.remove_node(node)

    logger.info("Sorting edges")
    sorted_edges = sorted(list(graph.edges(data=True)),
                          key=lambda e: e[2][metric])

    logger.info("Selecting shortest edges")
    for u, v, data in sorted_edges:
        if unselected.has_edge(u, v):
            graph.edges[(u, v)][selected_key] = True
            selected.add_edge(u, v)
            unselected.remove_edges_from(list(graph.out_edges(u)))
            if selected.in_degree(v) > 1:
                unselected.remove_edges_from(list(unselected.in_edges(v)))
