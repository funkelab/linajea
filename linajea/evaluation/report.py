class Report:
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
        self.fn_divisions = None
        self.fn_divs_no_connections = None
        self.fn_divs_unconnected_child = None
        self.fn_divs_unconnected_parent = None

        # METRICS
        self.precision = None
        self.recall = None
        self.fscore = None
        self.aeftl = None
        self.erl = None

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
        self.gt_tracks = gt_tracks
        self.rec_tracks = rec_tracks
        self.gt_matched_tracks = gt_matched_tracks
        self.rec_matched_tracks = rec_matched_tracks

    def set_edge_stats(
            self,
            gt_edges,
            rec_edges,
            matched_edges):
        self.gt_edges = gt_edges
        self.rec_edges = rec_edges
        self.matched_edges = matched_edges

    def set_division_stats(
            self,
            gt_divisions,
            rec_divisions):
        self.gt_divisions = gt_divisions
        self.rec_divisions = rec_divisions

    def set_fn_edges(self, fn_edges):
        self.fn_edges = len(fn_edges)
        self.fn_edge_list = fn_edges

    def set_fp_edges(self, num_fp_edges):
        self.fp_edges = num_fp_edges

    def set_identity_switches(self, identity_switches):
        self.identity_switches = len(identity_switches)
        self.identity_switch_gt_nodes = identity_switches

    def set_fn_divisions(
            self,
            fn_divs_no_connections,
            fn_divs_unconnected_child,
            fn_divs_unconnected_parent,
            tp_divs):
        self.fn_divs_no_connections = len(fn_divs_no_connections)
        self.fn_divs_unconnected_child = len(fn_divs_unconnected_child)
        self.fn_divs_unconnected_parent = len(fn_divs_unconnected_parent)
        self.fn_divisions = self.fn_divs_no_connections +\
            self.fn_divs_unconnected_child +\
            self.fn_divs_unconnected_parent

        self.no_connection_gt_nodes = fn_divs_no_connections
        self.unconnected_child_gt_nodes = fn_divs_unconnected_child
        self.unconnected_parent_gt_nodes = fn_divs_unconnected_parent
        self.tp_div_gt_nodes = tp_divs

    def set_fp_divisions(self, fp_divisions):
        self.fp_divisions = len(fp_divisions)
        self.fp_div_rec_nodes = fp_divisions

    def set_f_score(self):
        tp = self.matched_edges
        fp = self.fp_edges
        fn = self.fn_edges
        assert tp is not None, "Need matched edges for fscore calc"
        assert fp is not None, "Need fp edges for fscore calc"
        assert fn is not None, "Need fn edges for fscore calc"

        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)

        self.f_score = 2 * self.precision * self.recall / (
                self.precision + self.recall)

    def set_aeftl_and_erl(self, aeftl, erl):
        self.aeftl = aeftl
        self.erl = erl
