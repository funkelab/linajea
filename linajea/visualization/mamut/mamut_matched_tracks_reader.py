from __future__ import print_function, division, absolute_import
import logging
from .mamut_reader import MamutReader

logger = logging.getLogger(__name__)


class MamutMatchedTracksReader(MamutReader):
    def __init__(self):
        super(MamutReader, self).__init__()

    def read_data(self, data):
        (gt_tracks, rec_tracks, track_matches) = data
        matched_rec_tracks_indexes = [rec for gt, rec in track_matches
                                      if gt < len(gt_tracks)]
        matched_rec_tracks = []
        logger.debug(track_matches)
        logger.debug("Matched Rec Tracks: %s" % matched_rec_tracks_indexes)
        for index in matched_rec_tracks_indexes:
            matched_rec_tracks.append(rec_tracks[index])

        logger.info("Adding %d gt tracks" % len(gt_tracks))
        track_id = 0
        cells = []
        tracks = []
        for track in gt_tracks:
            track_cells, track = self.add_track(track, track_id, group=0)
            cells += track_cells
            tracks.append(track)
            track_id += 1

        logger.info("Adding %d matched rec tracks" % len(matched_rec_tracks))
        for track in matched_rec_tracks:
            track_cells, track = self.add_track(track, track_id, group=1)
            cells += track_cells
            tracks.append(track)
            track_id += 1

        return cells, tracks

    def add_track(self, track, track_id, group):
        if len(track.nodes) == 0:
            logger.info("Track has no nodes. Skipping")
            return [], []

        if len(track.edges) == 0:
            logger.info("Track has no edges. Skipping")
            return [], []

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
