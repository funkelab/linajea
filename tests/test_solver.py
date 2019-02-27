import linajea.tracking
import logging
import networkx as nx

logging.basicConfig(level=logging.INFO)
logging.getLogger('linajea.solver').setLevel(logging.DEBUG)

if __name__ == "__main__":

    #   x
    #  3|         /-4
    #  2|        /--3
    #  1|   0---1
    #  0|        \--2
    #    ------------------------------------ t
    #       0   1   2

    cells = [
        {'id': 0, 'position': (0, 0, 0, 1), 'frame': 0, 'score': 2.0},
        {'id': 1, 'position': (1, 0, 0, 1), 'frame': 1, 'score': 2.0},
        {'id': 2, 'position': (2, 0, 0, 0), 'frame': 2, 'score': 2.0},
        {'id': 3, 'position': (2, 0, 0, 2), 'frame': 2, 'score': 2.0},
        {'id': 4, 'position': (2, 0, 0, 3), 'frame': 2, 'score': 2.0},
    ]

    edges = [
        {'source': 1, 'target': 0, 'score': 1.0, 'distance': 0.0},
        {'source': 2, 'target': 1, 'score': 1.0, 'distance': 1.0},
        {'source': 3, 'target': 1, 'score': 1.0, 'distance': 1.0},
        {'source': 4, 'target': 1, 'score': 1.0, 'distance': 2.0},
    ]

    parameters = linajea.tracking.TrackingParameters()
    parameters.cost_appear = 1.0
    parameters.cost_disappear = 1.0
    parameters.cost_split = 0
    parameters.weight_distance_cost = 0.1
    parameters.weight_node_score = 1.0
    parameters.threshold_node_score = 0.0
    parameters.threshold_edge_score = 0.0

    graph = nx.DiGraph()
    graph.add_nodes_from([(cell['id'], cell) for cell in cells])
    graph.add_edges_from([(edge['source'], edge['target'], edge)
                          for edge in edges])
    linajea.tracking.track(graph, parameters, selected_key='selected')

    print("Selected edges:")
    for u, v, data in graph.edges(data=True):
        if data['selected']:
            print("%d -> %d" % (u, v))
