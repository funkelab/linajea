from __future__ import print_function, division, absolute_import
import os
import logging
from .mamut_xml_templates import (
        begin_template,
        alltracks_template,
        alltracks_end_template,
        filteredtracks_start_template,
        filteredtracks_template,
        filteredtracks_end_template,
        end_template,
        im_data_template,
        inframe_template,
        spot_template,
        allspots_template,
        allspots_end_template,
        track_template,
        edge_template,
        track_end_template,
        inframe_end_template,
        )

logger = logging.getLogger(__name__)


class MamutWriter:
    def __init__(self):
        self.cells_by_frame = {}
        self.max_cell_id = 0
        self.tracks = []

    def remap_cell_ids(self, cells, tracks):
        logger.info("Remapping cell ids")
        id_map = {}
        for cell in cells:
            old_id = cell['id']
            id_map[old_id] = self.max_cell_id
            cell['id'] = self.max_cell_id
            self.max_cell_id += 1
        for track in tracks:
            for edge in track['edges']:
                old_source = edge['source']
                old_target = edge['target']
                edge['source'] = id_map[old_source]
                edge['target'] = id_map[old_target]
        logging.debug("Cell id map: {}".format(id_map))
        if self.max_cell_id > 2000000000:
            logging.warn("Max ID after remapping %d is greater than 2 billion."
                         "Possible MaMuT error due to int overflow"
                         % self.max_cell_id)

    def add_data(self, mamut_reader, data):
        cells, tracks = mamut_reader.read_data(data)
        self.remap_cell_ids(cells, tracks)
        logger.info("Adding %d cells, %d tracks"
                    % (len(cells), len(tracks)))
        for cell in cells:
            pos = cell['position']
            time = pos[0]
            _id = cell['id']
            if time not in self.cells_by_frame:
                self.cells_by_frame[time] = []
            self.cells_by_frame[time].append(cell)
            if _id > self.max_cell_id:
                self.max_cell_id = _id

        self.tracks.extend(tracks)

    def write(self, raw_data_xml, output_xml, scale=1.0):
        if not self.cells_by_frame.keys():
            logger.error("No data to write. Exiting")
            exit(1)
        if not self.tracks:
            logger.info("No tracks present. Creating fake track"
                        " to avoid MaMuT error.")
            self.create_fake_track()
        with open(output_xml, 'w') as output:

            output.write(begin_template)

            self.cells_to_xml(output, scale)

            # Begin AllTracks.
            output.write(alltracks_template)
            logger.debug("Writing tracks {}".format(self.tracks))
            track_ids = [self.track_to_xml(track, _id,  output)
                         for _id, track in enumerate(self.tracks)]

            # End AllTracks.
            output.write(alltracks_end_template)

            # Filtered tracks.
            output.write(filteredtracks_start_template)
            for track_id in track_ids:
                if track_id is not None:
                    output.write(
                            filteredtracks_template.format(
                                t_id=track_id
                                )
                            )
            output.write(filteredtracks_end_template)

            # End XML file.
            folder, filename = os.path.split(os.path.abspath(raw_data_xml))
            if not folder:
                folder = os.path.dirname("./")
            output.write(end_template.format(
                image_data=im_data_template.format(
                    filename=filename,
                    folder=folder)))

    def cells_to_xml(self, output, scale=1.0):
        num_cells = 0
        for frame in self.cells_by_frame.keys():
            num_cells += len(self.cells_by_frame[frame])
        # Begin AllSpots.
        output.write(allspots_template.format(nspots=num_cells))

        # Loop through lists of spots.
        for t, cells in self.cells_by_frame.items():
            output.write(inframe_template.format(frame=t))
            for cell in cells:
                _, z, y, x = cell['position']
                score = cell['score'] if 'score' in cell else 0
                _id = cell['id']
                if 'name' in cell:
                    name = cell['name']
                else:
                    # backwards compatible
                    name = str(_id) + " SPOT_" + str(_id)
                output.write(
                    spot_template.format(
                        id=_id,
                        name=name,
                        frame=t,
                        quality=score,
                        z=z*scale,
                        y=y*scale,
                        x=x*scale))

            output.write(inframe_end_template)

        # End AllSpots.
        output.write(allspots_end_template)

    def track_to_xml(self, track, _id, output):
        logger.debug("Writing track {}".format(track))
        edges = track['edges']

        start = track['start']
        stop = track['stop']
        duration = stop - start + 1
        num_cells = track['num_cells']

        track_id = _id

        output.write(
            track_template.format(
                id=track_id,
                duration=duration,
                start=start, stop=stop,
                nspots=num_cells))
        if len(edges) == 0:
            logger.warn("No edges in track!")
        for edge in edges:
            source = edge['source']
            target = edge['target']
            score = edge['score']

            output.write(
                edge_template.format(
                    source_id=source, target_id=target,
                    score=score,
                    # Shouldn't the time be edge specific, not track?
                    time=start))
        output.write(track_end_template)

        return track_id

    def create_fake_track(self):
        start_time = list(self.cells_by_frame.keys())[0]
        end_time = start_time + 1
        start_point = self.cells_by_frame[start_time][0]
        end_point_position = [end_time] + start_point['position'][1:]
        end_point_id = self.max_cell_id + 1
        self.max_cell_id = end_point_id
        end_point = {'position': end_point_position,
                     'score': start_point['score'],
                     'id': end_point_id}
        if end_time not in self.cells_by_frame:
            self.cells_by_frame[end_time] = []
        self.cells_by_frame[end_time].append(end_point)

        edge = {'source': end_point_id,
                'target': start_point['id'],
                'score': 0}
        track = {'start': start_time,
                 'stop': end_time,
                 'num_cells': 2,
                 'id': 0,
                 'edges': [edge]}
        self.tracks.append(track)
