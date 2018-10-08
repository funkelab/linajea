import comatch
import logging
import numpy as np
import pylp

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
        positions_x = {}
        positions_y = {}

        for track_id, track_x in enumerate(tracks_x):

            frame_track_nodes_x = track_x.cells_by_frame(t)
            frame_nodes_x += frame_track_nodes_x
            labels_x.update({ n: track_id for n in frame_track_nodes_x })
            positions_x.update({
                n: track_x.nodes[n]['position'] for n in frame_track_nodes_x
            })

        for track_id, track_y in enumerate(tracks_y):

            frame_track_nodes_y = track_y.cells_by_frame(t)
            frame_nodes_y += frame_track_nodes_y
            labels_y.update({ n: track_id for n in frame_track_nodes_y })
            positions_y.update({
                n: track_y.nodes[n]['position'] for n in frame_track_nodes_y
            })

        nodes_x += frame_nodes_x
        nodes_y += frame_nodes_y

        for node_x in frame_nodes_x:
            for node_y in frame_nodes_y:

                distance = np.linalg.norm(
                    np.array(positions_x[node_x]) -
                    np.array(positions_y[node_y]))

                if distance <= matching_threshold:
                    edges_xy.append((node_x, node_y))
                    edge_costs.append(distance)

    return comatch.match_components(
        nodes_x, nodes_y,
        edges_xy,
        labels_x, labels_y,
        edge_costs=edge_costs,
        no_match_costs=2*matching_threshold)
