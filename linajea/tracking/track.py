from __future__ import absolute_import
from .solver import Solver
from .track_graph import TrackGraph
import logging

logger = logging.getLogger(__name__)

class TrackingParameters(object):

    def __init__(self):

        self.cost_appear = 0
        self.cost_disappear = 0
        self.cost_split = 0
        self.weight_distance_cost = 0
        self.threshold_node_score = 1
        self.threshold_edge_score = 1

def track(cells, edges, parameters):

    if len(cells) == 0:
        return

    logger.info("Creating track graph...")
    track_graph = TrackGraph()

    skipped_edges = 0

    for cell in cells:
        track_graph.add_cell(cell)

    for edge in edges:

        u, v = edge['source'], edge['target']

        if u in track_graph.nodes and v in track_graph.nodes:

            track_graph.add_cell_edge(edge)

        else:

            logger.debug(
                "Skipping edge %d -> %d, at least one node not in graph",
                u, v)
            skipped_edges += 1

    logger.info("Skipped %d edges without corresponding nodes", skipped_edges)

    logger.info("Creating solver...")
    solver = Solver(track_graph, parameters)

    logger.info("Solving...")
    solver.solve()

    for cell in cells:
        cell['selected'] = track_graph.nodes[cell['id']]['selected']
    for edge in edges:
        e = (edge['source'], edge['target'])
        if e in track_graph.edges:
            edge['selected'] = track_graph.edges[e]['selected']
