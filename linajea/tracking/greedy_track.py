from .track_graph import TrackGraph
import logging
import networkx as nx

logger = logging.getLogger(__name__)


def greedy_track(
        graph,
        selected_key,
        cell_indicator_threshold,
        metric='prediction_distance',
        frame_key='t'):
    if graph.number_of_nodes() == 0:
        logger.info("No nodes in graph - skipping solving step")
        return
    nx.set_edge_attributes(graph, False, selected_key)
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)
    start_frame, end_frame = track_graph.get_frames()

    # find "seed" cells in last frame
    seed_candidates = track_graph.cells_by_frame(end_frame)
    seeds = [node for node in seed_candidates
             if graph.nodes[node]['score'] > cell_indicator_threshold]
    while seeds:
        candidate_edges = []
        selected_nodes = []
        for seed in seeds:
            # edges that are within move_threshold (in pure euclidean, not PV)
            candidate_edges.extend(graph.out_edges(seed, data=True))

        logger.debug("Sorting edges in frame %d", graph.nodes[seeds[0]]['t'])
        sorted_edges = sorted(candidate_edges,
                              key=lambda e: e[2][metric])

        logger.debug("Selecting shortest edges")
        for u, v, data in sorted_edges:
            # check if child already has selected out edge
            already_selected = len([
                    u
                    for u, v, data in graph.out_edges(u, data=True)
                    if data[selected_key]]) > 0
            if already_selected:
                continue
            # check to make sure it's not overloading the parent
            two_children = len([
                    u
                    for u, v, data in graph.in_edges(v, data=True)
                    if data[selected_key]]) > 1
            if two_children:
                continue

            graph.edges[(u, v)][selected_key] = True
            selected_nodes.append(v)
        seeds = selected_nodes
