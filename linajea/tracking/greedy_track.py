from .track_graph import TrackGraph
import logging
import networkx as nx

logger = logging.getLogger(__name__)


def greedy_track(
        graph,
        selected_key,
        cell_indicator_threshold,
        metric='prediction_distance',
        frame_key='t',
        allow_new_tracks=True):
    if graph.number_of_nodes() == 0:
        logger.info("No nodes in graph - skipping solving step")
        return
    nx.set_edge_attributes(graph, False, selected_key)
    track_graph = TrackGraph(graph_data=graph,
                             frame_key=frame_key,
                             roi=graph.roi)
    start_frame, end_frame = track_graph.get_frames()

    selected_prev_nodes = []
    for frame in range(end_frame, start_frame + 1, -1):
        # find "seed" cells in frame
        seed_candidates = track_graph.cells_by_frame(frame)
        seeds = [node for node in seed_candidates
                 if graph.nodes[node]['score'] > cell_indicator_threshold]

        # use only new (not previously selected) nodes to seed new tracks
        seeds = [s for s in seeds if s not in selected_prev_nodes]

        if frame == end_frame:
            # in this special case, all seeds are treated as selected
            selected_prev_nodes = seeds
            seeds = []

        candidate_edges = []
        selected_next_nodes = []
        # pick the shortest edges greedily for the set of previously selected
        # nodes, with tree constraint (allow divisions)
        for selected_prev in selected_prev_nodes:
            candidate_edges.extend(graph.out_edges(selected_prev, data=True))

        logger.debug("Sorting edges in frame %d", frame)
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
            selected_next_nodes.append(v)

        if allow_new_tracks:
            # pick the shortest edges greedily for the set of new possible
            # tracks with one to one constraint (do not allow divisions)
            candidate_edges = []
            for seed in seeds:
                candidate_edges.extend(graph.out_edges(seed, data=True))
            sorted_edges = sorted(candidate_edges,
                                  key=lambda e: e[2][metric])

            for u, v, data in sorted_edges:
                # check if child already has selected out edge
                already_selected = len([
                        u
                        for u, v, data in graph.out_edges(u, data=True)
                        if data[selected_key]]) > 0
                if already_selected:
                    continue
                # check to make sure it's not overloading the parent
                one_child = len([
                        u
                        for u, v, data in graph.in_edges(v, data=True)
                        if data[selected_key]]) > 0
                if one_child:
                    continue

                graph.edges[(u, v)][selected_key] = True
                selected_next_nodes.append(v)

        selected_prev_nodes = selected_next_nodes
