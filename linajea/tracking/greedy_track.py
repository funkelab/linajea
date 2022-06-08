import logging
import networkx as nx
from linajea import CandidateDatabase
from daisy import Roi
from .track_graph import TrackGraph

logger = logging.getLogger(__name__)


def load_graph(
        cand_db,
        roi,
        selected_key):
    '''
    Args:
        cand_db (`linajea.CandidateDatabase`)
            Candidate Database from which to load graph

        roi (`daisy.Roi`)
            Roi to get data from.

        selected_key (`str`):
            Edge attribute used to store selection. Should be set to false by
            default

    Returns: A new nx.DiGraph with the data from cand_db in region roi.
    '''
    edge_attributes = ['distance', 'prediction_distance']
    graph = cand_db.get_graph(roi, edge_attrs=edge_attributes)
    # set selected key to false
    nx.set_edge_attributes(graph, False, selected_key)
    return graph


def greedy_track(
        graph=None,
        db_name=None,
        db_host=None,
        selected_key=None,
        cell_indicator_threshold=None,
        metric='prediction_distance',
        frame_key='t',
        allow_new_tracks=True,
        roi=None):
    if graph is None:
        cand_db = CandidateDatabase(db_name, db_host, 'r+')
        total_roi = cand_db.get_nodes_roi()
    else:
        if graph.number_of_nodes() == 0:
            logger.info("No nodes in graph - skipping solving step")
            return
        cand_db = None
        total_roi = graph.roi

    if roi is not None:
        total_roi = roi.intersect(total_roi)

    start_frame = total_roi.get_offset()[0]
    end_frame = start_frame + total_roi.get_shape()[0]
    logger.info("Tracking from frame %d to frame %d", end_frame, start_frame)
    if graph is None:
        step = 10
    else:
        step = end_frame - start_frame

    selected_prev_nodes = set()
    first = True
    for section_end in range(end_frame, start_frame, -1*step):
        section_begin = section_end - step
        frames_roi = Roi((section_begin, None, None, None),
                         (step, None, None, None))
        section_roi = total_roi.intersect(frames_roi)
        logger.info("Greedy tracking in section %s", str(section_roi))
        selected_prev_nodes = track_section(
                graph,
                cand_db,
                section_roi,
                selected_key,
                cell_indicator_threshold,
                selected_prev_nodes,
                metric=metric,
                frame_key=frame_key,
                allow_new_tracks=allow_new_tracks,
                first=first)
        first = False
        logger.debug("Done tracking in section %s", str(section_roi))


def track_section(
        graph,
        cand_db,
        roi,
        selected_key,
        cell_indicator_threshold,
        selected_prev_nodes,
        metric='prediction_distance',
        frame_key='t',
        allow_new_tracks=True,
        first=False):
    # this function solves this whole section and stores the result, and
    # returns the node ids in the preceeding frame (before roi!) that were
    # selected
    if graph is None:
        graph = load_graph(cand_db, roi, selected_key)
    else:
        nx.set_edge_attributes(graph, False, selected_key)
    track_graph = TrackGraph(graph_data=graph, frame_key=frame_key, roi=roi)
    start_frame = roi.get_offset()[0]
    end_frame = start_frame + roi.get_shape()[0] - 1

    for frame in range(end_frame, start_frame - 1, -1):
        logger.debug("Processing frame %d", frame)

        # find "seed" cells in frame
        seed_candidates = track_graph.cells_by_frame(frame)
        if len(selected_prev_nodes) > 0:
            assert [p in seed_candidates for p in selected_prev_nodes],\
                "previously selected nodes are not contained in current frame!"
        seeds = set([node for node in seed_candidates
                     if graph.nodes[node]['score'] > cell_indicator_threshold])
        logger.debug("Found %d potential seeds in frame %d", len(seeds), frame)

        # use only new (not previously selected) nodes to seed new tracks
        seeds = seeds - selected_prev_nodes
        logger.debug("Found %d seeds in frame %d", len(seeds), frame)

        if first and frame == end_frame:
            # in this special case, all seeds are treated as selected
            selected_prev_nodes = seeds
            seeds = set()

        candidate_edges = []
        selected_next_nodes = set()
        # pick the shortest edges greedily for the set of previously selected
        # nodes, with tree constraint (allow divisions)
        for selected_prev in selected_prev_nodes:
            candidate_edges.extend(graph.out_edges(selected_prev, data=True))

        logger.debug("Sorting %d candidate edges in frame %d",
                     len(candidate_edges), frame)
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
            selected_next_nodes.add(v)

        logger.debug("Selected %d continuing edges", len(selected_next_nodes))

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
                selected_next_nodes.add(v)
        logger.debug("Selected %d total nodes in next frame",
                     len(selected_next_nodes))

        selected_prev_nodes = selected_next_nodes
    logger.info("updating edges in roi %s" % roi)
    graph.update_edge_attrs(
                roi,
                attributes=selected_key)
    return selected_prev_nodes
