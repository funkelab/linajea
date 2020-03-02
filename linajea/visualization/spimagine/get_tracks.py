import daisy
import linajea.tracking
import linajea
import pymongo

def get_gt_tracks_for_roi(gt_db_name, mongo_url, roi):
    graph_provider = linajea.CandidateDatabase(gt_db_name, mongo_url)
    subgraph = graph_provider[roi]
    track_graph = linajea.tracking.TrackGraph(subgraph)
    tracks = track_graph.get_tracks()
    end_frame = roi.get_offset()[0] + roi.get_shape()[0] - 1
    one_d_tracks = []
    for track in tracks:
        for end_cell in track.get_cells_in_frame(end_frame):
            cell_positions = []
            current_cell = end_cell
            while current_cell is not None:
                current_data = track.nodes[current_cell]
                cell_positions.append([current_data[dim] for dim in ['t', 'z', 'y', 'x']])
                parent_edges = track.prev_edges(current_cell)
                if len(parent_edges) == 1:
                    current_cell = parent_edges[0][1]
                elif len(parent_edges) == 0:
                    current_cell = None
                else:
                    print("Error: Cell has two parents! Exiting")
                    return None
            one_d_tracks.append(cell_positions)
    print("Found %d tracks in roi %s" % (len(one_d_tracks), roi))
    return one_d_tracks


def get_node_ids_in_frame(gt_db_name, mongo_url, frame):
    graph_provider = linajea.CandidateDatabase(gt_db_name, mongo_url)
    roi = daisy.Roi((frame, 0, 0, 0), (1, 10e6, 10e6, 10e6))
    nodes = graph_provider.read_nodes(roi)
    node_ids = [node['id'] for node in nodes]
    return node_ids


def get_track_from_node(
        node_id,
        node_frame,
        gt_db_name,
        mongo_url,
        num_frames_before,
        num_frames_after=0):
    graph_provider = linajea.CandidateDatabase(gt_db_name, mongo_url)
    roi = daisy.Roi((node_frame - num_frames_before + 1, 0, 0, 0),
                    (num_frames_before + num_frames_after, 10e6, 10e6, 10e6))
    subgraph = graph_provider[roi]
    track_graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
    tracks = track_graph.get_tracks()
    for track in tracks:
        if track.has_node(node_id):
            cell_positions = []
            current_cell = node_id
            while True:
                current_data = track.nodes[current_cell]
                if 't' not in current_data:
                    break
                cell_positions.append([current_data[dim] for dim in ['t', 'z', 'y', 'x']])
                parent_edges = list(track.prev_edges(current_cell))
                if len(parent_edges) == 1:
                    current_cell = parent_edges[0][1]
                elif len(parent_edges) == 0:
                    break
                else:
                    print("Error: Cell has two parents! Exiting")
                    return None
            return cell_positions
    print("Did not find track with node %d in roi %s"
          % (node_id, roi))
    return None


def get_region_around_node(
        node_id,
        db_name,
        db_host,
        context,
        voxel_size):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    nodes = db['nodes']
    node = nodes.find_one({'id': node_id})
    if node is None:
        print("Did not fine node with %d in db %s" % (node_id, db_name))
    location = daisy.Coordinate([node[dim] for dim in ['t', 'z', 'y', 'x']])
    roi = daisy.Roi(location - context, context * 2)
    roi = roi.snap_to_grid(voxel_size)
    return roi
