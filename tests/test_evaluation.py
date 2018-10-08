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
