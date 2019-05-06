from __future__ import absolute_import
from .solver import Solver
from .track_graph import TrackGraph
import logging
import time

logger = logging.getLogger(__name__)


class TrackingParameters(object):

    def __init__(
            self,
            model_type='default',
            block_size=None,
            context=None,
            cost_appear=None,
            cost_disappear=None,
            cost_split=None,
            max_cell_move=None,
            threshold_node_score=None,
            weight_node_score=None,
            threshold_edge_score=None,
            weight_distance_cost=None,
            weight_prediction_distance_cost=None,
            **kwargs):

        # "default" for the models that predict parent vectors only
        # "nms" for the non-max suppression models
        assert model_type in ['default', 'nms']
        self.model_type = model_type

        # ALL MODEL TYPES:
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

        # ONLY DEFAULT MODEL:

        # how to weigh the Euclidean distance between cells for the costs of an
        # edge
        if model_type == 'default':
            assert weight_distance_cost is not None,\
                "Failed to specify weight_distance_cost"
        self.weight_distance_cost = weight_distance_cost

        # ONLY NMS MODEL:

        # how to weigh the Euclidean distance between the predicted position
        # and the actual position of cells for the costs of an edge
        if model_type == 'nms':
            assert weight_prediction_distance_cost is not None,\
                "Failed to specify weight_prediction_distance_cost"
        self.weight_prediction_distance_cost = weight_prediction_distance_cost


def track(graph, parameters, selected_key, frame_key='t'):
    # assuming graph is a daisy subgraph
    if graph.number_of_nodes() == 0:
        return

    logger.info("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)

    logger.info("Creating solver...")
    solver = Solver(track_graph, parameters, selected_key)

    logger.info("Solving...")
    start_time = time.time()
    solver.solve()
    logger.info("Solving ILP took %s seconds" % (time.time() - start_time))

    for cell, data in graph.nodes(data=True):
        data[selected_key] = track_graph.nodes[cell][selected_key]
    for u, v, data in graph.edges(data=True):
        if (u, v) in track_graph.edges:
            data[selected_key] = track_graph.edges[(u, v)][selected_key]
