"""Provides a class containing all different metrics for a single solution
"""
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class Report:
    """Object used to accumulate statistics for a solution

    Notes
    -----
    A filled Report object contains all computed metrics and statistics
    wrt to a single solution, such as number of gt/rec tracks, matched
    tracks, how many edges/division, how many errors of which kind,
    aggregated metrics such as precision, list of all edges/nodes
    involved in errors.
    """
    def __init__(self):
        # STATISTICS
        self.gt_tracks = None
        self.rec_tracks = None
        self.gt_matched_tracks = None
        self.rec_matched_tracks = None

        self.gt_edges = None
        self.rec_edges = None
        self.matched_edges = None

        self.gt_divisions = None
        self.rec_divisions = None

        # ERRORS
        self.fp_edges = None
        self.fn_edges = None
        self.identity_switches = None
        self.fp_divisions = None
        self.iso_fp_division = None
        self.fn_divisions = None
        self.iso_fn_division = None
        self.fn_divs_no_connections = None
        self.fn_divs_unconnected_child = None
        self.fn_divs_unconnected_parent = None

        # METRICS
        self.precision = None
        self.recall = None
        self.f_score = None
        self.aeftl = None
        self.erl = None
        self.correct_segments = None
        self.validation_score = None

        # FAILURE POINTS
        self.fn_edge_list = None
        self.identity_switch_gt_nodes = None
        self.fp_div_rec_nodes = None
        self.no_connection_gt_nodes = None
        self.unconnected_child_gt_nodes = None
        self.unconnected_parent_gt_nodes = None
        self.tp_div_gt_nodes = None

    def set_track_stats(
            self,
            gt_tracks,
            rec_tracks,
            gt_matched_tracks,
            rec_matched_tracks):
        '''
        Args:
            gt_tracks(int):
                number of gt tracks

            rec_tracks (int):
                number of reconstructed tracks

            gt_matched_tracks (int):
                number of gt tracks with at least one edge matched

            rec_matched_tracks (int):
                number of reconstructed tracks with at least
                one edge matched
        '''
        self.gt_tracks = gt_tracks
        self.rec_tracks = rec_tracks
        self.gt_matched_tracks = gt_matched_tracks
        self.rec_matched_tracks = rec_matched_tracks

    def set_edge_stats(
            self,
            gt_edges,
            rec_edges,
            matched_edges):
        '''
        Args:
            gt_edges (int):
                number of gt edges

            rec_edges (int):
                number of reconstructed edges

            matched edges (int):
                number of matched edges
        '''
        self.gt_edges = gt_edges
        self.rec_edges = rec_edges
        self.matched_edges = matched_edges

    def set_division_stats(
            self,
            gt_divisions,
            rec_divisions):
        '''
        Args:
            gt_divisions (int):
                number of gt divisions

            rec_divisions (int):
                number of reconstructed divisions
        '''
        self.gt_divisions = gt_divisions
        self.rec_divisions = rec_divisions

    def set_fn_edges(self, fn_edges):
        '''
        Args:
            fn_edges (list: (int, int)):
                The list of unmatched gt edges (u, v)
        '''
        self.fn_edges = len(fn_edges)
        self.fn_edge_list = [(int(s), int(t)) for s, t in fn_edges]

    def set_fp_edges(self, fp_edges):
        self.fp_edges = len(fp_edges)
        self.fp_edge_list = [(int(s), int(t)) for s, t in fp_edges]

    def set_identity_switches(self, identity_switches):
        '''
        Args:
            identity_switches (list of int):
                A list of gt node ids where identity switches occurred
        '''
        self.identity_switches = len(identity_switches)
        self.identity_switch_gt_nodes = [int(n) for n in identity_switches]

    def set_fn_divisions(
            self,
            fn_divs_no_connections,
            fn_divs_unconnected_child,
            fn_divs_unconnected_parent,
            tp_divs,
            fn_div_count_unconnected_parent):
        '''
        Args:
            fn_divs_... (list of int):
                Lists of gt node ids where different kinds of fn divisions
                occurred. See Evaluator.get_fn_divisions() for more details.

            tp_divs (list of int):
                List of gt node ids where the reconstruction correctly matches
                the division
        '''
        self.fn_divs_no_connections = len(fn_divs_no_connections)
        self.fn_divs_unconnected_child = len(fn_divs_unconnected_child)
        self.fn_divs_unconnected_parent = len(fn_divs_unconnected_parent)
        self.fn_divisions = self.fn_divs_no_connections + \
            self.fn_divs_unconnected_child
        if fn_div_count_unconnected_parent:
            self.fn_divisions += self.fn_divs_unconnected_parent

        self.no_connection_gt_nodes = [int(n) for n in fn_divs_no_connections]
        self.unconnected_child_gt_nodes = [
                int(n) for n in fn_divs_unconnected_child]
        self.unconnected_parent_gt_nodes = [
                int(n) for n in fn_divs_unconnected_parent]
        self.tp_div_gt_nodes = [int(n) for n in tp_divs]

    def set_fp_divisions(self, fp_divisions):
        '''
        Args:
            fp_divisions (list of int):
                List of reconstruction node ids where false positive
                divisions occurred. If dense gt, fp divs don't necessarily
                have a gt match, so we store the rec node ids.
        '''
        self.fp_divisions = len(fp_divisions)
        self.fp_div_rec_nodes = [int(n) for n in fp_divisions]

    def set_f_score(self):
        tp = self.matched_edges
        fp = self.fp_edges
        fn = self.fn_edges
        assert tp is not None, "Need matched edges for fscore calc"
        assert fp is not None, "Need fp edges for fscore calc"
        assert fn is not None, "Need fn edges for fscore calc"

        self.precision = 0 if tp + fp == 0 else tp / (tp + fp)
        self.recall = 0 if tp + fn == 0 else tp / (tp + fn)

        self.f_score = 0.0 if self.precision + self.recall == 0 else \
            2 * self.precision * self.recall / (
                self.precision + self.recall)

    def set_aeftl_and_erl(self, aeftl, erl):
        self.aeftl = aeftl
        self.erl = erl

    def set_validation_score(self, validation_score):
        self.validation_score = validation_score

    def set_iso_fn_divisions(self, iso_fn_div_nodes,
                             fn_div_count_unconnected_parent):
        """If used, remove fn divisions that are off by only a single
        frame from the list of false divisions and count them separately
        as iso(morphic) fn divisions

        Adapt fn edges/fp edges stats accordingly"""
        self.iso_fn_division = len(iso_fn_div_nodes)
        fn_div_gt_nodes = (self.no_connection_gt_nodes +
                           self.unconnected_child_gt_nodes)
        if fn_div_count_unconnected_parent:
            fn_div_gt_nodes += self.unconnected_parent_gt_nodes
        fn_div_gt_nodes = [f for f in fn_div_gt_nodes
                           if f not in iso_fn_div_nodes]
        self.fn_divisions = len(fn_div_gt_nodes)

        # remove fp/fn edges directly involved in iso fn div
        logger.debug("fn edges before iso_fn_div: %d", self.fn_edges)
        logger.debug("fp edges before iso_fn_div: %d", self.fp_edges)
        edges = self.fn_edge_list
        self.fn_edge_list = []
        for u, v in edges:
            found = False
            for d in iso_fn_div_nodes:
                if int(d) == u or int(d) == v:
                    found = True
            if not found:
                self.fn_edge_list.append((u, v))
        self.fn_edge_list = list(self.fn_edge_list)
        self.fn_edges = len(self.fn_edge_list)

        edges = self.fp_edge_list
        self.fp_edge_list = []
        for u, v in edges:
            found = False
            for d in iso_fn_div_nodes:
                if int(d) == u or int(d) == v:
                    found = True
            if not found:
                self.fp_edge_list.append((u, v))
        self.fp_edge_list = list(self.fp_edge_list)
        self.fp_edges = len(self.fp_edge_list)
        logger.debug("fn edges after iso_fn_div: %d", self.fn_edges)
        logger.debug("fp edges after iso_fn_div: %d", self.fp_edges)

    def set_iso_fp_divisions(self, iso_fp_div_nodes):
        """If used, remove fp divisions that are off by only a single
        frame from the list of false divisions and count them separately
        as iso(morphic) fp divisions

        Adapt fn edges/fp edges stats accordingly"""
        self.iso_fp_division = len(iso_fp_div_nodes)
        self.fp_div_rec_nodes = [f for f in self.fp_div_rec_nodes
                                 if f not in iso_fp_div_nodes]
        self.fp_divisions = len(self.fp_div_rec_nodes)

        # remove fp/fn edges directly involved in iso fp div
        logger.debug("fp edges before iso_fp_div: %d", self.fp_edges)
        logger.debug("fn edges before iso_fp_div: %d", self.fn_edges)
        edges = self.fp_edge_list
        self.fp_edge_list = []
        for u, v in edges:
            found = False
            for d in iso_fp_div_nodes:
                if int(d) == u or int(d) == v:
                    found = True
            if not found:
                self.fp_edge_list.append((u, v))
        self.fp_edge_list = list(self.fp_edge_list)
        self.fp_edges = len(self.fp_edge_list)

        edges = self.fn_edge_list
        self.fn_edge_list = []
        for u, v in edges:
            found = False
            for d in iso_fp_div_nodes:
                if int(d) == u or int(d) == v:
                    found = True
            if not found:
                self.fn_edge_list.append((u, v))
        self.fn_edge_list = list(self.fn_edge_list)
        self.fn_edges = len(self.fn_edge_list)
        logger.debug("fp edges after iso_fp_div: %d", self.fp_edges)
        logger.debug("fn edges after iso_fp_div: %d", self.fn_edges)


    def get_report(self):
        """Long report

        Returns
        -------
        dict
            Dictionary containing all attributes of report
        """
        return self.__dict__

    def get_short_report(self):
        """Short report without lists

        Returns
        -------
        dict
            Dictionary with all lists of false edges/nodes removed
        """
        report = deepcopy(self.__dict__)
        # STATISTICS
        del report['fn_edge_list']
        del report['fp_edge_list']
        del report['identity_switch_gt_nodes']
        del report['fp_div_rec_nodes']
        del report['no_connection_gt_nodes']
        del report['unconnected_child_gt_nodes']
        del report['unconnected_parent_gt_nodes']
        del report['tp_div_gt_nodes']
        del report['correct_segments']

        return report
