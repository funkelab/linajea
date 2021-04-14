from __future__ import print_function, division, absolute_import
import logging
from .mamut_reader import MamutReader
import linajea
import linajea.evaluation
from daisy import Roi

logger = logging.getLogger(__name__)


class MamutMatchedTracksReader(MamutReader):
    def __init__(self, db_host):
        super(MamutReader, self).__init__()
        self.db_host = db_host

    def read_data(self, data):
        candidate_db_name = data['db_name']
        start_frame, end_frame = data['frames']
        gt_db_name = data['gt_db_name']
        assert end_frame > start_frame
        roi = Roi((start_frame, 0, 0, 0),
                  (end_frame - start_frame, 1e10, 1e10, 1e10))
        if 'parameters_id' in data:
            try:
                int(data['parameters_id'])
                selected_key = 'selected_' + str(data['parameters_id'])
            except:
                selected_key = data['parameters_id']
        else:
            selected_key = None
        db = linajea.CandidateDatabase(
                candidate_db_name, self.db_host)
        db.selected_key = selected_key
        gt_db = linajea.CandidateDatabase(gt_db_name, self.db_host)

        print("Reading GT cells and edges in %s" % roi)
        gt_subgraph = gt_db[roi]
        gt_graph = linajea.tracking.TrackGraph(gt_subgraph, frame_key='t')
        gt_tracks = list(gt_graph.get_tracks())
        print("Found %d GT tracks" % len(gt_tracks))

        # tracks_to_xml(gt_cells, gt_tracks, 'linajea_gt.xml')

        print("Reading cells and edges in %s" % roi)
        subgraph = db.get_selected_graph(roi)
        graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
        tracks = list(graph.get_tracks())
        print("Found %d tracks" % len(tracks))
        
        if len(graph.nodes) == 0 or len(gt_graph.nodes) == 0:
            logger.info("Didn't find gt or reconstruction - returning")
            return [], []

        m = linajea.evaluation.match_edges(
            gt_graph, graph,
            matching_threshold=20)
        (edges_x, edges_y, edge_matches, edge_fps) = m
        matched_rec_tracks = []
        for track in tracks:
            for _, edge_index in edge_matches:
                edge = edges_y[edge_index]
                if track.has_edge(edge[0], edge[1]):
                    matched_rec_tracks.append(track)
                    break
        logger.debug("found %d matched rec tracks" % len(matched_rec_tracks))

        logger.info("Adding %d gt tracks" % len(gt_tracks))
        track_id = 0
        cells = []
        tracks = []
        for track in gt_tracks:
            result = self.add_track(track, track_id, group=0)
            print(result[0])
            if result is None or len(result[0]) == 0:
                continue
            track_cells, track = result
            cells += track_cells
            tracks.append(track)
            track_id += 1

        logger.info("Adding %d matched rec tracks" % len(matched_rec_tracks))
        for track in matched_rec_tracks:
            result = self.add_track(track, track_id, group=1)
            if result is None:
                continue
            track_cells, track = result
            cells += track_cells
            tracks.append(track)
            track_id += 1

        return cells, tracks

    def add_track(self, track, track_id, group):
        if len(track.nodes) == 0:
            logger.info("Track has no nodes. Skipping")
            return None

        if len(track.edges) == 0:
            logger.info("Track has no edges. Skipping")
            return None

        cells = []
        invalid_cells = []
        for _id, data in track.nodes(data=True):
            if _id == -1 or 't' not in data:
                logger.info("Cell %s with data %s is not valid. Skipping"
                            % (_id, data))
                invalid_cells.append(_id)
                continue
            position = [data['t'], data['z'], data['y'], data['x']]
            track.nodes[_id]['position'] = position
            score = group
            cells.append(self.create_cell(position, score, _id))

        track.remove_nodes_from(invalid_cells)

        track_edges = []
        for u, v, edge in track.edges(data=True):
            score = group
            track_edges.append(self.create_edge(u,
                                                v,
                                                score=score))
        start_time, end_time = track.get_frames()
        num_cells = len(track.nodes)
        track = self.create_track(start_time,
                                  end_time,
                                  num_cells,
                                  track_id,
                                  track_edges)
        return cells, track
