import logging
import math
import networkx as nx
from .report import Report
from .validation_metric import validation_score

logger = logging.getLogger(__name__)


class Evaluator:
    ''' A class for evaluating linajea results after matching.
    Creates a report with statistics, error counts, and locations
    of errors.

    Args:
        gt_track_graph (`linajea.TrackGraph`):
            The ground truth track graph

        rec_track_graph (`linajea.TrackGraph`):
            The reconstructed track graph

        edge_matches (list of ((int, int), (int, int))):
            List of edge matches, where each edge match
            is a tuple of (x_edge, y_edge), and each edge is a tuple
            of (source_id, target_id)

        unselected_potential_matches (int):
            The number of rec edges within the matching radius
            that were not matched to a gt edge (to use as a
            proxy for edge fps if sparse annotations)

        sparse (bool):
            True if the ground truth data is sparse, false if it is
            dense. Changes how edge and division false positives
            are counted. Defaults to true.
    '''
    def __init__(
            self,
            gt_track_graph,
            rec_track_graph,
            edge_matches,
            unselected_potential_matches,
            sparse=True,
            validation_score=True,
            window_size=50,
            ):
        self.report = Report()

        self.gt_track_graph = gt_track_graph
        self.rec_track_graph = rec_track_graph
        self.edge_matches = edge_matches
        self.unselected_potential_matches = unselected_potential_matches
        self.sparse = sparse
        self.validation_score = validation_score
        self.window_size = window_size

        # get tracks
        self.gt_tracks = gt_track_graph.get_tracks()
        self.rec_tracks = rec_track_graph.get_tracks()
        logger.debug("Found %d gt tracks and %d rec tracks"
                     % (len(self.gt_tracks), len(self.rec_tracks)))
        self.matched_track_ids = self.__get_track_matches()

        # get track statistics
        rec_matched_tracks = set()
        for y_matches in self.matched_track_ids.values():
            rec_matched_tracks.update(y_matches)
        self.report.set_track_stats(
                len(self.gt_tracks),
                len(self.rec_tracks),
                len(self.matched_track_ids.keys()),
                len(rec_matched_tracks))

        # get edge statistics
        self.report.set_edge_stats(
                gt_track_graph.number_of_edges(),
                rec_track_graph.number_of_edges(),
                len(self.edge_matches))

        # get division statistics
        self.gt_parents = [node for node, degree in gt_track_graph.in_degree()
                           if degree == 2]
        self.rec_parents = [node for node, degree
                            in rec_track_graph.in_degree() if degree == 2]
        self.report.set_division_stats(len(self.gt_parents),
                                       len(self.rec_parents))

        # get data structures needed for evaluation
        self.gt_edges_to_rec_edges = {}
        self.rec_edges_to_gt_edges = {}
        for gt_edge, rec_edge in self.edge_matches:
            self.gt_edges_to_rec_edges[gt_edge] = rec_edge
            self.rec_edges_to_gt_edges[rec_edge] = gt_edge

    def evaluate(self):
        self.get_fp_edges()
        self.get_fn_edges()
        self.get_identity_switches()
        self.get_fp_divisions()
        self.get_fn_divisions()
        self.get_f_score()
        self.get_aeftl_and_erl()
        self.get_perfect_segments(self.window_size)
        if self.validation_score:
            self.get_validation_score()
        return self.report

    def get_fp_edges(self):
        ''' Store the number of fp edges in self.report.
        If sparse, this is the number of unselected potential matches.
        If dense, this is the total number of unmatched rec edges.
        '''
        if self.sparse:
            fp_edges = self.unselected_potential_matches
        else:
            fp_edges = self.report.rec_edges - self.report.matched_edges
        self.report.set_fp_edges(fp_edges)

    def get_fn_edges(self):
        num_fn_edges = self.report.gt_edges - self.report.matched_edges

        matched_edges = set([match[0] for match in self.edge_matches])
        gt_edges = set(self.gt_track_graph.edges)
        fn_edges = list(gt_edges - matched_edges)
        assert len(fn_edges) == num_fn_edges, "List of fn edges "\
            "has %d edges, but calculated %d fn edges"\
            % (len(fn_edges), num_fn_edges)

        self.report.set_fn_edges(fn_edges)

        for fn_edge in fn_edges:
            self.gt_track_graph.edges[fn_edge]['FN'] = True

    def get_identity_switches(self):
        ''' Store the number of identity switches and the ids
        of the gt nodes where the IS occurs in self.report.

        Will loop through all non-division gt_cells, see if prev_edge and
        next_edge have matches, and if these edge matches match the same
        rec cell to the gt cell. Ignores division nodes.
        '''
        is_nodes = []
        for gt_cell in self.gt_track_graph.nodes():
            next_edges = list(self.gt_track_graph.next_edges(gt_cell))
            if len(next_edges) != 1:
                # ignore parent nodes and nodes without any children
                continue

            prev_edges = list(self.gt_track_graph.prev_edges(gt_cell))
            if len(prev_edges) == 0:
                continue
            assert len(prev_edges) == 1,\
                "GT cell has more than one previous edge (%s)" % prev_edges
            prev_edge = prev_edges[0]
            if prev_edge not in self.gt_edges_to_rec_edges:
                continue
            prev_edge_match = self.gt_edges_to_rec_edges[prev_edge]

            next_edge = next_edges[0]
            if next_edge not in self.gt_edges_to_rec_edges:
                continue
            next_edge_match = self.gt_edges_to_rec_edges[next_edge]
            if next_edge_match[1] != prev_edge_match[0]:
                logger.debug("Prev edge match %s source does not match"
                             " next edge match target %s: identity switch"
                             % (prev_edge_match, next_edge_match))
                is_nodes.append(gt_cell)

        self.report.set_identity_switches(is_nodes)
        for is_node in is_nodes:
            self.gt_track_graph.nodes[is_node]['IS'] = True

    def get_fp_divisions(self):
        ''' Store the number of fp divisions and the rec node ids
        in self.report. If sparse, ignore rec divisions where no adjacent
        edges (next or previous) match to ground truth, and if the
        previous match is the end of a gt track.
        '''

        fp_div_nodes = []
        for rec_parent in self.rec_parents:
            next_edges = self.rec_track_graph.next_edges(rec_parent)
            assert len(next_edges) == 2,\
                "Parent cell must have two next edges (got %s)" % next_edges
            next_edge_matches = [self.rec_edges_to_gt_edges.get(next_edge)
                                 for next_edge in next_edges]
            c1_t = next_edge_matches[0][1] if next_edge_matches[0]\
                is not None else None
            c2_t = next_edge_matches[1][1] if next_edge_matches[1]\
                is not None else None
            logger.debug("c1_t: %s" % str(c1_t))
            logger.debug("c2_t: %s" % str(c2_t))

            if next_edge_matches.count(None) == 2 and self.sparse:
                logger.debug("Neither child edge matched, and sparse is true.")
                prev_edges = list(self.rec_track_graph.prev_edges(rec_parent))
                if len(prev_edges) == 0:
                    logger.debug("No edges match and sparse is true: continue")
                    continue
                assert len(prev_edges) == 1
                prev_edge = prev_edges[0]
                prev_edge_match = self.rec_edges_to_gt_edges.get(prev_edge)
                if prev_edge_match is None:
                    logger.debug("No edges match and sparse is true: continue")
                    continue
                if len(self.gt_track_graph.next_edges(prev_edge_match)) == 0:
                    logger.debug("Div occurs after end of gt track: continue")
                    continue
                # in the case that neither child edge matches but the parent
                # does (in the middle of a gt track) - this is a fp edge
                # and will be taken care of below

            if not self.__div_match_node_equality(c1_t, c2_t):
                logger.debug("FP division at rec node %d" % rec_parent)
                fp_div_nodes.append(rec_parent)
                if c1_t is not None:
                    self.gt_track_graph.nodes[c1_t]['FP_D'] = True
                if c2_t is not None:
                    self.gt_track_graph.nodes[c2_t]['FP_D'] = True
        self.report.set_fp_divisions(fp_div_nodes)

    def __div_match_node_equality(self, n1, n2):
        if n1 is None or n2 is None:
            return False
        return n1 == n2

    def get_fn_divisions(self):
        ''' Store the number of fn divisions in
        self.error_metrics['num_fn_divisions']
        FN divisions are split into 3 categories. Consider node equality
        as defined in __div_match_equality (both nodes must exist and
        have the same id).
        The categories are defined by equalities between:
        prev_s = the source node of the edge matched to the gt prev edge
        c1_t, c2_t = the target nodes of the edges matched to the gt next edges
        (c1_t and c2_t are interchangeable)
        If there is no edge match for a given gt edge,
        the associated node is None

        1) No Connections
        self.error_metrics['num_divs_no_connections']
        None of the surrounding tracklets are connected.
        prev_s != c1_t != c2_t

        2) One Unconnected Child
        self.error_metrics['num_divs_one_unconnected_child']
        if parent track continues to one child and not the other
        (prev_s = c1_t or prev_s = c2_t) and c1_t != c2_t

        3) Unconnected Parent
        self.error_metrics['num_divs_unconnected_parent']
        if child tracks connect to each other but not to the parent
        c1_t = c2_t and prev_s != c1_t

        In the special case that there is no gt prev edge, we only consider
        c1_t and c2_t. If c1_t = c2_t, no error.
        If c1_t != c2_t, no connections.
        '''
        fn_div_no_connection_nodes = []
        fn_div_unconnected_child_nodes = []
        fn_div_unconnected_parent_nodes = []
        tp_div_nodes = []

        for gt_parent in self.gt_parents:
            next_edges = self.gt_track_graph.next_edges(gt_parent)
            assert len(next_edges) == 2,\
                ("Parent cell must have two next edges (got %s)"
                 % next_edges)
            next_edge_matches = [self.gt_edges_to_rec_edges.get(gt_edge)
                                 for gt_edge in next_edges]

            c1_t = next_edge_matches[0][1] if next_edge_matches[0]\
                is not None else None
            c2_t = next_edge_matches[1][1] if next_edge_matches[1]\
                is not None else None
            logger.debug("c1_t: %s" % str(c1_t))
            logger.debug("c2_t: %s" % str(c2_t))

            prev_edges = list(self.gt_track_graph.prev_edges(gt_parent))
            assert len(prev_edges) <= 1,\
                ("Parent cell must have one or zero prev edges, got %s"
                 % len(prev_edges))
            if len(prev_edges) == 0:
                # special case where gt division has no prev edge
                logger.warning("GT Division at node %d has no prev edge"
                               % gt_parent)
                if self.__div_match_node_equality(c1_t, c2_t):
                    # TP - children are connected
                    logger.debug("TP division - no gt parent")
                    tp_div_nodes.append(gt_parent)
                    continue
                else:
                    # FN - No connections
                    logger.debug("FN - no connections")
                    fn_div_no_connection_nodes.append(gt_parent)
                    continue
            prev_edge = prev_edges[0]
            prev_edge_match = self.gt_edges_to_rec_edges.get(prev_edge)
            prev_s = prev_edge_match[0] if prev_edge_match\
                is not None else None
            logger.debug("prev_s: %s" % str(prev_s))

            if self.__div_match_node_equality(c1_t, c2_t):
                if self.__div_match_node_equality(c1_t, prev_s):
                    # TP
                    logger.debug("TP div")
                    tp_div_nodes.append(gt_parent)
                else:
                    # FN - Unconnected parent
                    logger.debug("FN div - unconnected parent")
                    fn_div_unconnected_parent_nodes.append(gt_parent)
            else:
                if self.__div_match_node_equality(c1_t, prev_s)\
                        or self.__div_match_node_equality(c2_t, prev_s):
                    # FN - one unconnected child
                    logger.debug("FN div - one unconnected child")
                    fn_div_unconnected_child_nodes.append(gt_parent)
                else:
                    # FN - no connections
                    logger.debug("FN div - no connections")
                    fn_div_no_connection_nodes.append(gt_parent)
        self.report.set_fn_divisions(fn_div_no_connection_nodes,
                                     fn_div_unconnected_child_nodes,
                                     fn_div_unconnected_parent_nodes,
                                     tp_div_nodes)

    def get_f_score(self):
        self.report.set_f_score()

    def get_aeftl_and_erl(self):
        ''' Store the AEFTL and ERL in self.report
        '''
        logger.info("Getting AEFTL and ERL")

        rec_matched_edges = self.rec_edges_to_gt_edges.keys()
        rec_matched_graph = self.rec_track_graph.edge_subgraph(
                rec_matched_edges).copy()
        max_node_id = max(list(rec_matched_graph.nodes) + [0])
        # split at fp_divisions
        for fp_div_node in self.report.fp_div_rec_nodes:
            if fp_div_node in rec_matched_graph:
                prev_edges = list(rec_matched_graph.prev_edges(fp_div_node))
                next_edges = list(rec_matched_graph.next_edges(fp_div_node))
            else:
                continue
            if len(prev_edges) == 0 or len(next_edges) == 0:
                continue
            for prev_edge in prev_edges:
                rec_matched_graph.remove_edge(prev_edge[0], prev_edge[1])
                new_node_id = max_node_id + 1
                rec_matched_graph.add_node(new_node_id)
                rec_matched_graph.add_edge(new_node_id, prev_edge[1])
                max_node_id += 1
        # split into connected components
        segments = [rec_matched_graph.subgraph(node_set).copy() for node_set in
                    nx.weakly_connected_components(rec_matched_graph)]
        logger.debug("Segment node sets: %s"
                     % list(nx.weakly_connected_components(
                         rec_matched_graph)))
        segment_lengths = [g.number_of_edges()
                           for g in segments if g.number_of_edges() > 0]

        logger.debug("Found segment lengths %s" % segment_lengths)
        aeftl = 0 if not len(segment_lengths) else \
            float(sum(segment_lengths)) / len(segment_lengths)
        erl = 0 if not self.gt_track_graph.number_of_edges() else \
            sum(map(
                lambda b: math.pow(b, 2),
                segment_lengths
            )) / self.gt_track_graph.number_of_edges()
        self.report.set_aeftl_and_erl(aeftl, erl)

    def get_perfect_segments(self, window_size):
        ''' Compute the percent of gt track segments that are correctly
        reconstructed using a sliding window of each size from 1 to t. Store
        the dictionary from window size to (# correct, total #) in self.report
        '''
        logger.info("Getting perfect segments")
        total_segments = {}
        correct_segments = {}
        for i in range(1, window_size + 1):
            total_segments[i] = 0
            correct_segments[i] = 0

        for gt_track in self.gt_tracks:
            for start_node in gt_track.nodes():
                start_edges = gt_track.next_edges(start_node)
                for start_edge in start_edges:
                    frames = 1
                    correct = True
                    current_nodes = [start_node]
                    next_edges = [start_edge]
                    while len(next_edges) > 0:
                        if correct:
                            # check current node and next edge
                            for current_node in current_nodes:
                                if current_node != start_node:
                                    if 'IS' in self.gt_track_graph.nodes[current_node] or\
                                            'FP_D' in self.gt_track_graph.nodes[current_node]:
                                        correct = False
                            for next_edge in next_edges:
                                if 'FN' in self.gt_track_graph.get_edge_data(*next_edge):
                                    correct = False
                        # update segment counts
                        total_segments[frames] += 1
                        if correct:
                            correct_segments[frames] += 1
                        # update loop variables
                        frames += 1
                        current_nodes = [u for u, v in next_edges]
                        next_edges = gt_track.next_edges(current_nodes)
                        if frames > window_size:
                            break

        result = {}
        for i in range(1, window_size + 1):
            result[str(i)] = (correct_segments[i], total_segments[i])
        self.report.correct_segments = result

    @staticmethod
    def check_track_validity(track_graph):
        # 0 or 1 parent per node
        out_degrees = [d for _, d in track_graph.out_degree()]
        logger.debug("Out degrees: %s" % out_degrees)
        assert max(out_degrees) <= 1,\
            "Track has a node with %d > 1 parent" % max(out_degrees)

        # <=2 children per node
        in_degrees = [d for _, d in track_graph.in_degree()]
        max_index = in_degrees.index(max(in_degrees))
        assert max(in_degrees) <= 2,\
            "Track has node %d with %d > 2 children" %\
            (list(track_graph.nodes())[max_index], max(in_degrees))

    def __get_track_matches(self):
        self.edges_to_track_id_rec = {}
        self.edges_to_track_id_gt = {}
        track_ids_gt_to_rec = {}

        for index, track in enumerate(self.rec_tracks):
            for edge in track.edges():
                self.edges_to_track_id_rec[edge] = index
        for index, track in enumerate(self.gt_tracks):
            for edge in track.edges():
                self.edges_to_track_id_gt[edge] = index

        for gt_edge, rec_edge in self.edge_matches:
            gt_track_id = self.edges_to_track_id_gt[gt_edge]
            rec_track_id = self.edges_to_track_id_rec[rec_edge]
            track_ids_gt_to_rec.setdefault(gt_track_id, set())
            track_ids_gt_to_rec[gt_track_id].add(rec_track_id)
        return track_ids_gt_to_rec

    def get_validation_score(self):
        vald_score = validation_score(
                self.gt_track_graph,
                self.rec_track_graph)
        self.report.set_validation_score(vald_score)
