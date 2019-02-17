from __future__ import absolute_import
from .solver import Solver
from .track_graph import TrackGraph
import logging

logger = logging.getLogger(__name__)

class TrackingParameters(object):

    def __init__(self):

        # "default" for the models that predict parent vectors only
        # "nms" for the non-max suppression models
        self.model_type = 'default'

        # ALL MODEL TYPES:

        # track costs:
        self.cost_appear = 0
        self.cost_disappear = 0
        self.cost_split = 0

        # node costs:

        # nodes with scores below this threshold will have a positive cost,
        # above this threshold a negative cost
        self.threshold_node_score = 1

        # scaling factor after the conversion to costs above
        self.weight_node_score = 0

        # edge costs:

        # how to weigh the Euclidean distance between cells for the costs of an
        # edge
        self.weight_distance_cost = 0

        # ONLY DEFAULT MODEL:

        # similar to node costs, determines when a cost is positive/negative
        self.threshold_edge_score = 1

        # ONLY NMS MODEL:

        # how to weigh the Euclidean distance between the predicted position and
        # the actual position of cells for the costs of an edge
        self.weight_prediction_distance_cost = 0


def track(graph, parameters, selected_key, frame_key='frame'):

    if graph.number_of_nodes() == 0:
        return

    logger.info("Creating track graph...")
    track_graph = TrackGraph(graph_data=graph, frame_key=frame_key)

    logger.info("Creating solver...")
    solver = Solver(track_graph, parameters, selected_key)

    logger.info("Solving...")
    solver.solve()

    for cell, data in graph.nodes(data=True):
        data[selected_key] = track_graph.nodes[cell][selected_key]
    for u, v, data in graph.edges(data=True):
        if (u, v) in track_graph.edges:
            data[selected_key] = track_graph.edges[(u, v)][selected_key]
