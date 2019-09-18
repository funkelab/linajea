import logging
from collections import deque
import math

logger = logging.getLogger(__name__)


class Evaluator:
    ''' A class for evaluating linajea results after matching.
    Takes flags to indicate which metrics to compute.
    Statistics about the gt and reconstrution stored in a dict
    in self.stats. Error metrics in self.error_metrics.
    Specific locations in the gt or reconstruction where errors
    occurr in self.error_details.

    Args:
        x_track_graph (`linajea.TrackGraph`):
            The ground truth track graph

        y_track_graph (`linajea.TrackGraph`):
            The reconstructed track graph

        edge_matches (list of ((int, int), (int, int))):
            List of edge matches, where each edge match
            is a tuple of (x_edge, y_edge), and each edge is a tuple
            of (source_id, target_id)

        unselected_potential_matches (int):
            The number of rec edges within the matching radius
            that were not matched to a gt edge (to use as a
            proxy for edge fps if sparse annotations)
    '''
    def __init__(
            self,
            x_track_graph,
            y_track_graph,
            edge_matches,
            unselected_potential_matches,
            ):
        self.stats = {}
        self.error_metrics = {}
        self.error_details = {}

        self.x_track_graph = x_track_graph
        self.y_track_graph = y_track_graph
        self.edge_matches = edge_matches
        self.unselected_potential_matches = unselected_potential_matches

        # get tracks
        self.x_tracks = x_track_graph.get_tracks()
        self.y_tracks = y_track_graph.get_tracks()
        logger.debug("Found %d gt tracks and %d rec tracks"
                     % (len(self.x_tracks), len(self.y_tracks)))
        self.matched_track_ids = self.__get_track_matches()

        # get statistics
        self._get_track_stats()
        self._get_edge_stats()
        self._get_division_stats()

    def evaluate(
            self,
            sparse=True,
            fn_division_edges=False,
            f_score=False,
            aeftl=False,
            error_details=False,
            **kwargs):
        ''' Compute error metrics and details.

        Args:
            sparse (bool):
                If True, assume the ground truth is sparse. If False,
                assume the ground truth is dense. Calculate edge and div
                false positives accordingly. Defaults to True.

            fn_division_edges (bool):
                If True, calculate the number of fn edges that are in
                divisions. Regardless, fn_edges includes all fn_edges,
                including those with divisions. To separate the division
                fn edges out, you will have to set this to true and then
                subtract from fn_edges externally. Defaults to False.

            f_score (bool):
                Flag indicating whether or not to calculate edge precision,
                recall, and f-score. Defaults to False

            aeftl (bool):
                Flag indicating whether or not to calculate the AEFTL and ERL.
                Defaults to False.

            error_details (bool):
                Flag indicating whether or not to store node_ids of errors.
                If True, will store the gt edge (source_id, target_id) of all
                fn edges, the x node_id of all identity switches, the y parent
                node id of all fp divisions, and the x parent node id of all
                fn and tp divisions.

            kwargs:
                Any extra arguments, to be ignored
        '''
        self.get_fn_edges(error_details=error_details)
        self.get_fp_edges(sparse=sparse)
        self.get_identity_switches(error_details=error_details)
        self.get_fp_divisions(sparse=sparse, error_details=error_details)
        self.get_fn_divisions(count_fn_edges=fn_division_edges,
                              error_details=error_details)
        if f_score:
            self.get_f_score()
        if aeftl:
            self.get_aeftl_and_erl()

    def get_fn_edges(self, error_details=False):
        ''' Store the number of false negative edges in
        self.error_metrics['num_fn_edges'], and optionally store the
        (source_id, target_id) of all fn edges in
        self.error_dtails['fn_edges']
        '''
        self.error_metrics['num_fn_edges'] = self.stats['num_gt_edges']\
            - self.stats['num_matched_edges']
        if error_details:
            matched_edges = set([match[0] for match in self.edge_matches])
            gt_edges = set(self.x_track_graph.edges)
            self.error_details['fn_edges'] = gt_edges - matched_edges

    def get_fp_edges(self, sparse=True):
        ''' Store the number of fp edges in self.error_metrics['num_fp_edges'].
        If sparse, this is the number of unselected potential matches.
        If dense, this is the number of unmatched rec edges.
        '''
        if sparse:
            self.error_metrics['num_fp_edges'] =\
                self.unselected_potential_matches
        else:
            self.error_metrics['num_fp_edges'] = self.stats['num_rec_edges']\
                - self.stats['num_matched_edges']

    def get_identity_switches(self, error_details=False):
        ''' Store the number of identity switches in
        self.error_metrics['identity_switches'] and optionally
        store the gt node ids of all locations of identity switches
        in self.error_details['identity_switches']

        Will loop through all gt_cells, for all pairs of prev_edge (<=1)
        and next_edge (<=2), see if both edges have matches, and if
        these edge matches match the same rec cell to the gt cell.
        '''
        num_is = 0
        if error_details:
            is_nodes = []
        x_edges_to_y_edges = self.__get_x_edges_to_y_edges()
        for gt_cell in self.x_track_graph.nodes():
            prev_edges = list(self.x_track_graph.prev_edges(gt_cell))
            if len(prev_edges) == 0:
                continue
            assert len(prev_edges) == 1,\
                "GT cell has more than one previous edge (%s)" % prev_edges
            prev_edge = prev_edges[0]
            if prev_edge not in x_edges_to_y_edges:
                continue
            prev_edge_match = x_edges_to_y_edges[prev_edge]

            next_edges = self.x_track_graph.next_edges(gt_cell)
            if len(next_edges) == 0:
                continue
            assert len(next_edges) <= 2,\
                "GT cell has more than two next edges (%s)" % next_edges
            for next_edge in next_edges:
                if next_edge not in x_edges_to_y_edges:
                    continue
                next_edge_match = x_edges_to_y_edges[next_edge]
                if next_edge_match[1] != prev_edge_match[0]:
                    logger.debug("Prev edge match %s source does not match"
                                 " next edge match target %s: identity switch"
                                 % (prev_edge_match, next_edge_match))
                    num_is += 1
                    if error_details:
                        is_nodes.append(gt_cell)
        self.error_metrics['identity_switches'] = num_is
        if error_details:
            self.error_details['identity_switches'] = is_nodes

    def get_fp_divisions(self, sparse=True, error_details=False):
        ''' Store the number of fp divisions in
        self.error_metrics['num_fp_divisions'], and optionally store
        the y parent ids of all fp divisions in
        error_details['fp_divisions']. If sparse, ignore
        y divisions where no adjacent edges (next or previous) match
        to ground truth.

        For every division in rec tracks, determine if next edges match
        to gt, and if matches have same gt target node
        '''
        try:
            self.y_parents
        except AttributeError:
            self.get_division_stats()

        y_edges_to_x_edges = self.__get_y_edges_to_x_edges()
        fp_divisions = 0
        if error_details:
            fp_div_nodes = []
        for y_parent in self.y_parents:
            next_edges = self.y_track_graph.next_edges(y_parent)
            assert len(next_edges) == 2,\
                "Parent cell must have two next edges (got %s)" % next_edges
            next_edge_matches = [y_edges_to_x_edges.get(next_edge)
                                 for next_edge in next_edges]
            if next_edge_matches.count(None) == 2 and sparse:
                logger.debug("Neither child edge matched, and sparse is true")
                prev_edges = self.y_track_graph.prev_edges(y_parent)
                if len(prev_edges) == 0:
                    logger.debug("No parent edge to match, and sparse is true."
                                 " No fp div")
                    continue
                assert len(prev_edges) == 1,\
                    ("y parent has more than one previous edge (%s)"
                     % prev_edges)
                prev_edge = list(prev_edges)[0]
                if prev_edge not in y_edges_to_x_edges:
                    logger.debug("Previous edge also has no match, and "
                                 "sparse is true - ignoring this division")
                    continue
            if None in next_edge_matches:
                logger.debug("At least one rec division edge had no match. "
                             "FP division")
                fp_divisions += 1
                if error_details:
                    fp_div_nodes.append(y_parent)
                continue
            if next_edge_matches[0][1] != next_edge_matches[1][1]:
                logger.debug("gt matches for division edges do not "
                             "have same target (%s, %s). FP division."
                             % (next_edge_matches[0], next_edge_matches[1]))
                fp_divisions += 1
                if error_details:
                    fp_div_nodes.append(y_parent)
        self.error_metrics['num_fp_divisions'] = fp_divisions
        if error_details:
            self.error_details['fp_divisions'] = fp_div_nodes

    def get_fn_divisions(self, count_fn_edges=True, error_details=False):
        ''' Store the number of fn divisions in
        self.error_metrics['num_fn_divisions']. Any gt division that is
        not recreated exactly by the reconstruction is fn (even if
        all edges are matched, since they could be matched to different tracks)

        For every division parent node in gt tracks, determine if next edges
        have matches, and if matches point to the same target node
        '''
        try:
            self.x_parents
        except AttributeError:
            self.get_division_stats()

        x_edges_to_y_edges = self.__get_x_edges_to_y_edges()
        fn_divisions = 0
        if error_details:
            fn_div_nodes = []
            tp_div_nodes = []
        if count_fn_edges:
            fn_division_edges = 0

        for x_parent in self.x_parents:
            next_edges = self.x_track_graph.next_edges(x_parent)
            assert len(next_edges) == 2,\
                ("Parent cell must have two next edges (got %s)"
                 % next_edges)
            next_edge_matches = [x_edges_to_y_edges.get(x_edge)
                                 for x_edge in next_edges]
            if None in next_edge_matches:
                logger.debug("At least one gt division edge had no match. "
                             "FN division")
                fn_divisions += 1
                if error_details:
                    fn_div_nodes.append(x_parent)
                if count_fn_edges:
                    fn_division_edges += next_edge_matches.count(None)
                continue
            if next_edge_matches[0][1] != next_edge_matches[1][1]:
                logger.debug("rec matches for division edges do not have"
                             " same target (%s, %s). FN division."
                             % (next_edge_matches[0], next_edge_matches[1]))
                fn_divisions += 1
                if error_details:
                    fn_div_nodes.append(x_parent)
            else:
                # tp division
                if error_details:
                    tp_div_nodes.append(x_parent)
        self.error_metrics['num_fn_divisions'] = fn_divisions
        if count_fn_edges:
            self.error_metrics['num_fn_division_edges'] = fn_division_edges
        if error_details:
            self.error_details['fn_divisions'] = fn_div_nodes
            self.error_details['tp_divisions'] = tp_div_nodes

    def get_f_score(self):
        tp = self.stats['num_matched_edges']
        fp = self.error_metrics['num_fp_edges']
        fn = self.error_metrics['num_fn_edges']
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        self.error_metrics['precision'] = precision
        self.error_metrics['recall'] = recall
        self.error_metrics['f_score'] = 2 * precision * recall / (
                precision + recall)

    def get_aeftl_and_erl(self):
        ''' Store the AEFTL and ERL in self.error_metrics['aeftl'/'erl']
        THIS HAS A BUG AND POTENTIALLY DOUBLE COUNTS BRANCHES
        AFTER A DIVISION - BEWARE!
        '''

        logger.warn("AEFTL/ERL CALCULATION IS BUGGY")
        logger.info("Getting AEFTL and ERL")

        reconstruction_lengths = []
        x_edges_to_y_edges = self.__get_x_edges_to_y_edges()

        logger.info("Getting segment lengths")
        for x_track in self.x_tracks:
            logger.debug("Getting segments for track with nodes %s"
                         % list(x_track.nodes()))

            segment_lengths = []
            start_of_gt_track = True
            start_frame = x_track.get_frames()[0]
            start_cells = deque(x_track.cells_by_frame(start_frame))
            assert len(start_cells) == 1, "%s cells in start frame %d"\
                % (start_cells, start_frame)

            while len(start_cells) > 0:
                start_cell = start_cells.popleft()
                if start_cell not in x_track.nodes:
                    continue
                next_edges = x_track.next_edges(start_cell)
                next_edges = deque([(edge, None) for edge in next_edges])
                length = 0

                while len(next_edges) > 0:
                    next_edge, prev_cell_match = next_edges.pop()
                    source, target = next_edge

                    if next_edge not in x_edges_to_y_edges:
                        # false negative, no next edge in this segment
                        # add source (later cell) to start_cells
                        start_cells.append(source)
                        continue

                    edge_match = x_edges_to_y_edges[next_edge]
                    source_match, target_match = edge_match

                    if prev_cell_match is not None and\
                            prev_cell_match != target_match:
                        # identity switch - no next edge in this segment
                        # add target (earlier cell) to start_cells
                        # TODO: if target is a parent, this causes
                        # one of the child branches to be processed twice
                        start_cells.append(target)
                        continue

                    next_edges_in_matched_track = list(
                            self.y_track_graph.next_edges(target_match))
                    next_edges_in_gt_track = list(
                            x_track.next_edges(target))

                    if len(next_edges_in_matched_track) > 1:
                        fp_division = False
                        # division in matched track
                        if len(next_edges_in_gt_track) != 2:
                            # no div in gt track -> fp division
                            fp_division = True
                        else:
                            assert next_edge in next_edges_in_gt_track
                            next_edges_in_gt_track.remove(next_edge)
                            other_edge_x = next_edges_in_gt_track[0]
                            other_edge_match = x_edges_to_y_edges.get(other_edge_x)
                            if other_edge_match not in next_edges_in_matched_track:
                                # other edges in div don't match -> fp division
                                fp_division = True

                        if fp_division and\
                                (start_of_gt_track or prev_cell_match is not None):
                            # false positive division, no next edge in this segment
                            # add target (earlier cell) to start_cells
                            start_cells.append(target)
                            continue

                    # edge continues segment
                    # add one to length of segment
                    # add next edges to queue for this segment
                    length += 1
                    continuing_edges = x_track.next_edges(source)
                    for cont_edge in continuing_edges:
                        next_edges.append((cont_edge, source_match))

                # no more edges in this segment
                if length > 0:
                    segment_lengths.append(length)
                start_of_gt_track = False

            logger.debug("Found segment lengths %s" % segment_lengths)
            # add track stats to overall
            reconstruction_lengths.extend(segment_lengths)

        logger.debug("Segment lengths: %s" % reconstruction_lengths)
        self.error_metrics['aeftl'] = float(
            sum(reconstruction_lengths)) / len(reconstruction_lengths)
        self.error_metrics['erl'] = sum(map(
            lambda b: math.pow(b, 2),
            reconstruction_lengths
            )) / self.x_track_graph.number_of_edges()

    def _get_track_stats(self):
        self.stats['num_gt_tracks'] = len(self.x_tracks)
        self.stats['num_rec_tracks'] = len(self.y_tracks)
        self.stats['num_gt_matched_tracks'] = len(
                self.matched_track_ids.keys())
        rec_matched_tracks = set()
        for y_matches in self.matched_track_ids.values():
            rec_matched_tracks.update(y_matches)
        self.stats['num_rec_matched_tracks'] = len(rec_matched_tracks)

    def _get_edge_stats(self):
        self.stats['num_gt_edges'] = self.x_track_graph.number_of_edges()
        self.stats['num_rec_edges'] = self.y_track_graph.number_of_edges()
        self.stats['num_matched_edges'] = len(self.edge_matches)

    def _get_division_stats(self):
        x_node_degrees = self.x_track_graph.in_degree()
        max_x_node_degree = max([v for _, v in x_node_degrees])
        logger.debug("Max degree for gt tracks: %d"
                     % max_x_node_degree)
        assert max_x_node_degree <= 2,\
            ("Max in degree should be less than 2, "
             "got %d in gt track graph"
             % max_x_node_degree)
        self.x_parents = [node for node, degree in x_node_degrees
                          if degree == 2]
        logger.debug("gt parent nodes: %s" % self.x_parents)
        self.stats['num_gt_divisions'] = len(self.x_parents)
        y_node_degrees = self.y_track_graph.in_degree()
        max_y_node_degree = max([v for _, v in y_node_degrees])
        logger.debug("Max degree for rec tracks: %d"
                     % max_y_node_degree)
        assert max_y_node_degree <= 2,\
            ("Max in degree should be less than 2, "
             "got %d in rec track graph"
             % max_y_node_degree)
        self.y_parents = [node for node, degree in y_node_degrees
                          if degree == 2]
        logger.debug("rec parent nodes: %s" % self.y_parents)
        self.stats['num_rec_divisions'] = len(self.y_parents)

    def __repr__(self):

        lines = []
        lines.append("STATISTICS")
        for k, v in self.stats.items():
            lines.append(k + "\t\t" + str(v))
        lines.append("")
        lines.append("ERROR METRICS")
        for k, v in self.error_metrics.items():
            lines.append(k + "\t\t" + str(v))
        return "\n".join(lines)

    def __get_track_matches(self):
        self.edges_to_track_id_y = {}
        self.edges_to_track_id_x = {}
        track_ids_x_to_y = {}

        for index, track in enumerate(self.y_tracks):
            for edge in track.edges():
                self.edges_to_track_id_y[edge] = index
        for index, track in enumerate(self.x_tracks):
            for edge in track.edges():
                self.edges_to_track_id_x[edge] = index

        for x_edge, y_edge in self.edge_matches:
            x_track_id = self.edges_to_track_id_x[x_edge]
            y_track_id = self.edges_to_track_id_y[y_edge]
            track_ids_x_to_y.setdefault(x_track_id, set())
            track_ids_x_to_y[x_track_id].add(y_track_id)
        return track_ids_x_to_y

    def __get_x_edges_to_y_edges(self):
        try:
            return self.x_edges_to_y_edges
        except AttributeError:
            self.x_edges_to_y_edges = {}
            for x_edge, y_edge in self.edge_matches:
                self.x_edges_to_y_edges[x_edge] = y_edge
            return self.x_edges_to_y_edges

    def __get_y_edges_to_x_edges(self):
        try:
            return self.y_edges_to_x_edges
        except AttributeError:
            self.y_edges_to_x_edges = {}
            for x_edge, y_edge in self.edge_matches:
                self.y_edges_to_x_edges[y_edge] = x_edge
            return self.y_edges_to_x_edges
