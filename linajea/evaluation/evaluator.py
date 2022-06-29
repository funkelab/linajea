"""Defines evaluator object
"""
from copy import deepcopy
import itertools
import logging
import math
import networkx as nx
from .report import Report
from .validation_metric import validation_score

logger = logging.getLogger(__name__)


class Evaluator:
    ''' A class for evaluating linajea results after matching.

    Takes two graphs and precomputed matched edges.
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
            sparse,
            validation_score,
            window_size,
            ignore_one_off_div_errors,
            fn_div_count_unconnected_parent
            ):
        self.report = Report()

        self.gt_track_graph = gt_track_graph
        self.rec_track_graph = rec_track_graph
        self.edge_matches = edge_matches
        self.unselected_potential_matches = unselected_potential_matches
        self.sparse = sparse
        self.validation_score = validation_score
        self.window_size = window_size
        self.ignore_one_off_div_errors = ignore_one_off_div_errors
        self.fn_div_count_unconnected_parent = fn_div_count_unconnected_parent

        # get tracks
        self.gt_tracks = gt_track_graph.get_tracks()
        self.rec_tracks = rec_track_graph.get_tracks()
        logger.debug("Found %d gt tracks and %d rec tracks"
                     % (len(self.gt_tracks), len(self.rec_tracks)))
        self.matched_track_ids = self._get_track_matches()

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
        if self.ignore_one_off_div_errors:
            self.get_div_topology_stats()

        return self.report

    def get_fp_edges(self):
        ''' Store the number of fp edges in self.report.
        If sparse, this is the number of unselected potential matches.
        If dense, this is the total number of unmatched rec edges.
        '''
        if self.sparse:
            num_fp_edges = self.unselected_potential_matches
        else:
            num_fp_edges = self.report.rec_edges - self.report.matched_edges

        matched_edges = set([match[1] for match in self.edge_matches])
        rec_edges = set(self.rec_track_graph.edges)
        fp_edges = list(rec_edges - matched_edges)
        assert len(fp_edges) == num_fp_edges, "List of fp edges "\
            "has %d edges, but calculated %d fp edges"\
            % (len(fp_edges), num_fp_edges)

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

        self.fp_div_nodes = []
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
                node = [self.rec_track_graph.nodes[rec_parent]['t'],
                        self.rec_track_graph.nodes[rec_parent]['z'],
                        self.rec_track_graph.nodes[rec_parent]['y'],
                        self.rec_track_graph.nodes[rec_parent]['x']]
                logger.debug("FP division at rec node %d %s %s" % (
                    rec_parent, node, next_edge_matches))
                self.fp_div_nodes.append(rec_parent)
                if c1_t is not None:
                    self.gt_track_graph.nodes[c1_t]['FP_D'] = True
                if c2_t is not None:
                    self.gt_track_graph.nodes[c2_t]['FP_D'] = True
        self.report.set_fp_divisions(self.fp_div_nodes)

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
        self.fn_div_no_connection_nodes = []
        self.fn_div_unconnected_child_nodes = []
        self.fn_div_unconnected_parent_nodes = []
        self.tp_div_nodes = []

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
                    self.tp_div_nodes.append(gt_parent)
                    continue
                else:
                    # FN - No connections
                    logger.debug("FN - no connections")
                    self.fn_div_no_connection_nodes.append(gt_parent)
                    self.gt_track_graph.nodes[gt_parent]['FN_D'] = True
                    continue
            prev_edge = prev_edges[0]
            prev_edge_match = self.gt_edges_to_rec_edges.get(prev_edge)
            prev_s = prev_edge_match[0] if prev_edge_match\
                is not None else None
            logger.debug("prev_s: %s" % str(prev_s))

            is_tp_div = False
            if self.__div_match_node_equality(c1_t, c2_t):
                if self.__div_match_node_equality(c1_t, prev_s) or \
                   not self.fn_div_count_unconnected_parent:
                    # TP
                    logger.debug("TP div")
                    self.tp_div_nodes.append(gt_parent)
                    is_tp_div = True
                if not self.__div_match_node_equality(c1_t, prev_s):
                    # FN - Unconnected parent
                    logger.debug("FN div - unconnected parent")
                    self.fn_div_unconnected_parent_nodes.append(gt_parent)
            else:
                if self.__div_match_node_equality(c1_t, prev_s)\
                        or self.__div_match_node_equality(c2_t, prev_s):
                    # FN - one unconnected child
                    logger.debug("FN div - one unconnected child")
                    self.fn_div_unconnected_child_nodes.append(gt_parent)
                    self.gt_track_graph.nodes[gt_parent]['FN_D'] = True
                else:
                    # FN - no connections
                    logger.debug("FN div - no connections")
                    self.fn_div_no_connection_nodes.append(gt_parent)
                    self.gt_track_graph.nodes[gt_parent]['FN_D'] = True

            if not is_tp_div:
                node = [self.gt_track_graph.nodes[gt_parent]['t'],
                        self.gt_track_graph.nodes[gt_parent]['z'],
                        self.gt_track_graph.nodes[gt_parent]['y'],
                        self.gt_track_graph.nodes[gt_parent]['x']]
                logger.debug("FN division at gt node %d %s %s" % (
                    gt_parent, node, next_edge_matches))
        self.report.set_fn_divisions(self.fn_div_no_connection_nodes,
                                     self.fn_div_unconnected_child_nodes,
                                     self.fn_div_unconnected_parent_nodes,
                                     self.tp_div_nodes,
                                     self.fn_div_count_unconnected_parent)

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
                                    if ('IS' in self.gt_track_graph.nodes[current_node] or
                                        'FP_D' in self.gt_track_graph.nodes[current_node] or
                                        'FN_D' in self.gt_track_graph.nodes[current_node]):
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

    def _get_track_matches(self):
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
                deepcopy(self.gt_track_graph),
                deepcopy(self.rec_track_graph))
        self.report.set_validation_score(vald_score)

    def get_div_topology_stats(self):
        """Look for `isomorphic` division errors

        For each division error, check if there is one 1 frame earlier
        or later. If yes, do not count it as an error. Only called if
        self.ignore_one_off_div_errors is set.
        """
        self.iso_fn_div_nodes = []
        for fn_div_node in itertools.chain(
                self.fn_div_no_connection_nodes,
                self.fn_div_unconnected_child_nodes,
                self.fn_div_unconnected_parent_nodes):

            gt_tmp_grph, rec_tmp_grph = self._get_local_graphs(
                fn_div_node,
                self.gt_track_graph, self.rec_track_graph)

            if len(gt_tmp_grph.nodes()) == 0 or len(rec_tmp_grph) == 0:
                continue
            if nx.is_isomorphic(gt_tmp_grph, rec_tmp_grph):
                fp_div_node = None
                for node, degree in rec_tmp_grph.degree():
                    if degree == 3:
                        fp_div_node = node
                logger.debug("found isomorphic fn division: %d/%s",
                             fp_div_node, fn_div_node)
                self.iso_fn_div_nodes.append(fn_div_node)
            else:
                logger.debug("not-isomorphic fn division: %d", fn_div_node)
        self.iso_fp_div_nodes = []
        for fp_div_node in self.fp_div_nodes:
            fp_div_node = int(fp_div_node)
            rec_tmp_grph, gt_tmp_grph = self._get_local_graphs(
                fp_div_node,
                self.rec_track_graph, self.gt_track_graph, rec_to_gt=True)
            if len(gt_tmp_grph.nodes()) == 0 or len(rec_tmp_grph) == 0:
                continue
            if nx.is_isomorphic(gt_tmp_grph, rec_tmp_grph):
                fn_div_node = None
                for node, degree in gt_tmp_grph.degree():
                    if degree == 3:
                        fn_div_node = node
                logger.debug("found isomorphic fp division: %d/%s",
                             fp_div_node, fn_div_node)
                self.iso_fp_div_nodes.append(fp_div_node)
            else:
                logger.debug("not-isomorphic fp division: %d", fp_div_node)

        self.report.set_iso_fn_divisions(self.iso_fn_div_nodes,
                                         self.fn_div_count_unconnected_parent)
        self.report.set_iso_fp_divisions(self.iso_fp_div_nodes)

    def _get_local_graphs(self, div_node, g1, g2, rec_to_gt=False):

        g1_nodes = []
        try:
            for n1 in g1.successors(div_node):
                g1_nodes.append(n1)
                for n2 in g1.successors(n1):
                    g1_nodes.append(n2)
                for n2 in g1.predecessors(n1):
                    g1_nodes.append(n2)
            for n1 in g1.predecessors(div_node):
                g1_nodes.append(n1)
                for n2 in g1.successors(n1):
                    g1_nodes.append(n2)
                for n2 in g1.predecessors(n1):
                    g1_nodes.append(n2)
        except:
            raise RuntimeError("Overlooked edge case in _get_local_graph?")

        prev_edge = list(g1.prev_edges(div_node))
        prev_edge = prev_edge[0] if len(prev_edge) > 0 else None
        next_edges = list(g1.next_edges(div_node))
        prev_edge_match = None
        next_edge_match = None
        if not rec_to_gt:
            if prev_edge is not None:
                prev_edge_match = self.gt_edges_to_rec_edges.get(prev_edge)
            if prev_edge_match is None:
                for next_edge in next_edges:
                    next_edge_match = self.gt_edges_to_rec_edges.get(next_edge)
                    if next_edge_match is not None:
                        break
        else:
            if prev_edge is not None:
                prev_edge_match = self.rec_edges_to_gt_edges.get(prev_edge)
            if prev_edge_match is None:
                for next_edge in next_edges:
                    next_edge_match = self.rec_edges_to_gt_edges.get(next_edge)
                    if next_edge_match is not None:
                        break

        g2_nodes = []
        if prev_edge_match is not None or next_edge_match is not None:
            if prev_edge_match is not None:
                div_node_match = prev_edge_match[0]
            else:
                div_node_match = next_edge_match[0]

            for n1 in g2.successors(div_node_match):
                g2_nodes.append(n1)
                for n2 in g2.successors(n1):
                    g2_nodes.append(n2)
                for n2 in g2.predecessors(n1):
                    g2_nodes.append(n2)
            for n1 in g2.predecessors(div_node_match):
                g2_nodes.append(n1)
                for n2 in g2.successors(n1):
                    g2_nodes.append(n2)
                for n2 in g2.predecessors(n1):
                    g2_nodes.append(n2)

        g1_tmp_grph = g1.subgraph(g1_nodes).to_undirected()
        g2_tmp_grph = g2.subgraph(g2_nodes).to_undirected()

        if len(g1_tmp_grph.nodes()) != 0 and len(g2_tmp_grph.nodes()) != 0:
            g1_tmp_grph = _contract(g1_tmp_grph)
            g2_tmp_grph = _contract(g2_tmp_grph)
        return g1_tmp_grph, g2_tmp_grph


def _contract(g):
    """
    Contract chains of neighbouring vertices with degree 2 into one hypernode.
    Arguments:
    ----------
    g -- networkx.Graph instance
    Returns:
    --------
    h -- networkx.Graph instance
        the contracted graph
    hypernode_to_nodes -- dict: int hypernode -> [v1, v2, ..., vn]
        dictionary mapping hypernodes to nodes

    Notes
    -----
    Based on https://stackoverflow.com/a/52329262
    """

    # create subgraph of all nodes with degree 2
    is_chain = [node for node, degree in g.degree() if degree <= 2]
    chains = g.subgraph(is_chain)

    # contract connected components (which should be chains of variable length)
    # into single node
    components = [chains.subgraph(c).copy()
                  for c in nx.connected_components(chains)]
    hypernode = max(g.nodes()) + 1
    hypernodes = []
    hyperedges = []
    hypernode_to_nodes = dict()
    false_alarms = []
    for component in components:
        if component.number_of_nodes() > 1:

            hypernodes.append(hypernode)
            vs = [node for node in component.nodes()]
            hypernode_to_nodes[hypernode] = vs

            # create new edges from the neighbours of the chain ends to the
            # hypernode
            component_edges = [e for e in component.edges()]
            for v, w in [e for e in g.edges(vs)
                         if not ((e in component_edges) or
                                 (e[::-1] in component_edges))]:
                if v in component:
                    hyperedges.append([hypernode, w])
                else:
                    hyperedges.append([v, hypernode])

            hypernode += 1

        # nothing to collapse as there is only a single node in component:
        else:
            false_alarms.extend([node for node in component.nodes()])

    # initialise new graph with all other nodes
    not_chain = [node for node in g.nodes() if node not in is_chain]
    h = g.subgraph(not_chain + false_alarms).copy()
    h.add_nodes_from(hypernodes)
    h.add_edges_from(hyperedges)

    return h
