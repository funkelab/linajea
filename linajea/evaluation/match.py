import comatch
import logging
import numpy as np
import pylp
import scipy.sparse
import scipy.spatial

logger = logging.getLogger(__name__)

def match_tracks(tracks_x, tracks_y, matching_threshold):

    begin = min([ t.get_frames()[0] for t in tracks_x + tracks_y ])
    end = max([ t.get_frames()[1] for t in tracks_x + tracks_y ]) + 1

    nodes_x = []
    nodes_y = []
    edges_xy = []
    edge_costs = []
    labels_x = {}
    labels_y = {}

    for t in range(begin, end):

        frame_nodes_x = []
        frame_nodes_y = []
        positions_x = []
        positions_y = []

        # get all nodes and their positions in X and Y of the current frame

        for track_id, track_x in enumerate(tracks_x):

            frame_track_nodes_x = track_x.cells_by_frame(t)

            labels_x.update({ n: track_id for n in frame_track_nodes_x })

            frame_nodes_x += frame_track_nodes_x
            positions_x += [
                track_x.nodes[n]['position'][1:]
                for n in frame_track_nodes_x
            ]

        for track_id, track_y in enumerate(tracks_y):

            frame_track_nodes_y = track_y.cells_by_frame(t)

            labels_y.update({ n: track_id for n in frame_track_nodes_y })

            frame_nodes_y += frame_track_nodes_y
            positions_y += [
                track_y.nodes[n]['position'][1:]
                for n in frame_track_nodes_y
            ]

        nodes_x += frame_nodes_x
        nodes_y += frame_nodes_y

        if len(frame_nodes_x) == 0 or len(frame_nodes_y) == 0:
            continue

        # get all matching edges and distances between X and Y of the current
        # frame

        kd_tree_x = scipy.spatial.KDTree(positions_x)
        kd_tree_y = scipy.spatial.KDTree(positions_y)

        neighbors_xy = kd_tree_x.query_ball_tree(kd_tree_y, matching_threshold)
        for i, js in enumerate(neighbors_xy):
            node_x = frame_nodes_x[i]
            for j in js:
                node_y = frame_nodes_y[j]
                distance = np.linalg.norm(
                    np.array(positions_x[i]) -
                    np.array(positions_y[j]))
                edges_xy.append((node_x, node_y))
                edge_costs.append(distance)

    return comatch.match_components(
        nodes_x, nodes_y,
        edges_xy,
        labels_x, labels_y,
        edge_costs=edge_costs,
        no_match_costs=2*matching_threshold)
