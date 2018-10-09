import daisy
import linajea.tracking
import linajea.evaluation
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('linajea.evaluation').setLevel(logging.DEBUG)

if __name__ == "__main__":

    x = linajea.tracking.TrackGraph()
    x.add_cell({'id': 1, 'position': [0, 0, 0, 0], 'frame': 0})
    x.add_cell({'id': 2, 'position': [1, 0, 0, 0], 'frame': 1})
    x.add_cell({'id': 3, 'position': [2, 0, 0, 0], 'frame': 2})
    x.add_cell_edge({'source': 2, 'target': 1})

    y = linajea.tracking.TrackGraph()
    y.add_cell({'id': 1, 'position': [0, 0, 0, 0], 'frame': 0})
    y.add_cell({'id': 2, 'position': [1, 0, 0, 0], 'frame': 1})
    y.add_cell({'id': 3, 'position': [2, 0, 0, 0], 'frame': 2})
    y.add_cell({'id': 10, 'position': [0, 1, 0, 0], 'frame': 0})
    y.add_cell({'id': 20, 'position': [1, 1, 0, 0], 'frame': 1})
    y.add_cell_edge({'source': 2, 'target': 1})
    y.add_cell_edge({'source': 20, 'target': 10})

    tracks_x = x.get_tracks()
    tracks_y = y.get_tracks()

    print("Found %d tracks in x"%len(tracks_x))
    print("Found %d tracks in y"%len(tracks_y))

    print(linajea.evaluation.match_tracks(
        tracks_x, tracks_y,
        matching_threshold=2))

    # real data test

    db_name = 'linajea_setup01_400000_140521_default'
    gt_db_name = 'linajea_140521_gt'
    roi = daisy.Roi((250, 0, 0, 0), (8, 1e10, 1e10, 1e10))

    db = linajea.CandidateDatabase(db_name, '10.40.4.51')
    gt_db = linajea.CandidateDatabase('linajea_140521_gt', '10.40.4.51')

    print("Reading GT cells and edges in %s"%roi)
    gt_cells = gt_db.read_nodes(roi)
    gt_edges = gt_db.read_edges(roi)
    gt_graph = linajea.tracking.TrackGraph(gt_cells, gt_edges)
    gt_tracks = list(gt_graph.get_tracks())
    print("Found %d GT tracks"%len(gt_tracks))

    print("Reading cells and edges in %s"%roi)
    cells = db.read_nodes(roi)
    edges = db.read_edges(roi)
    graph = linajea.tracking.TrackGraph(cells, edges)
    tracks = list(graph.get_tracks(require_selected=True))
    print("Found %d tracks"%len(tracks))

    scores = linajea.evaluation.evaluate(
        gt_tracks, tracks,
        matching_threshold=25)

    print(scores)
