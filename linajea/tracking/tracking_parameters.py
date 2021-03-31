class TrackingParameters(object):

    def __init__(
            self,
            block_size=None,
            context=None,
            track_cost=None,
            max_cell_move=None,
            selection_constant=None,
            weight_node_score=None,
            weight_edge_score=None,
            division_constant=1,
            weight_division=0,
            weight_child=0,
            weight_continuation=0,
            version=None,
            **kwargs):

        # block size and context
        assert block_size is not None, "Failed to specify block_size"
        self.block_size = block_size
        assert context is not None, "Failed to specify context"
        self.context = context

        # track costs:
        assert track_cost is not None, "Failed to specify track_cost"
        self.track_cost = track_cost

        # max_cell_move
        # nodes within this distance to the block boundary will not pay
        # the appear and disappear costs
        # (Should be < 1/2 the context in z/x/y)
        assert max_cell_move is not None, "Failed to specify max_cell_move"
        self.max_cell_move = max_cell_move

        assert selection_constant is not None,\
            "Failed to specify selection_constant"
        self.selection_constant = selection_constant

        # scaling factors
        assert weight_node_score is not None,\
            "Failed to specify weight_node_score"
        self.weight_node_score = weight_node_score

        assert weight_edge_score is not None,\
            "Failed to specify weight_edge_score"
        self.weight_edge_score = weight_edge_score

        # Cell cycle
        self.division_constant = division_constant
        self.weight_division = weight_division
        self.weight_child = weight_child
        self.weight_continuation = weight_continuation

        # version control
        self.version = version


class NMTrackingParameters(object):

    def __init__(
            self,
            block_size=None,
            context=None, cost_appear=None,
            cost_disappear=None,
            cost_split=None,
            max_cell_move=None,
            threshold_node_score=None,
            weight_node_score=None,
            threshold_edge_score=None,
            weight_prediction_distance_cost=None,
            version=None,
            **kwargs):

        # block size and context
        assert block_size is not None, "Failed to specify block_size"
        self.block_size = block_size
        assert context is not None, "Failed to specify context"
        self.context = context

        # track costs:
        assert cost_appear is not None, "Failed to specify cost_appear"
        self.cost_appear = cost_appear
        assert cost_disappear is not None, "Failed to specify cost_disappear"
        self.cost_disappear = cost_disappear
        assert cost_split is not None, "Failed to specify cost_split"
        self.cost_split = cost_split

        # max_cell_move
        # nodes within this distance to the block boundary will not pay
        # the appear and disappear costs
        # (Should be < 1/2 the context in z/x/y)
        assert max_cell_move is not None, "Failed to specify max_cell_move"
        self.max_cell_move = max_cell_move

        # node costs:

        # nodes with scores below this threshold will have a positive cost,
        # above this threshold a negative cost
        assert threshold_node_score is not None,\
            "Failed to specify threshold_node_score"
        self.threshold_node_score = threshold_node_score

        # scaling factor after the conversion to costs above
        assert weight_node_score is not None,\
            "Failed to specify weight_node_score"
        self.weight_node_score = weight_node_score

        # edge costs:

        # similar to node costs, determines when a cost is positive/negative
        assert threshold_edge_score is not None,\
            "Failed to specify threshold_edge_score"
        self.threshold_edge_score = threshold_edge_score

        # how to weigh the Euclidean distance between the predicted position
        # and the actual position of cells for the costs of an edge
        assert weight_prediction_distance_cost is not None,\
            "Failed to specify weight_prediction_distance_cost"
        self.weight_prediction_distance_cost = weight_prediction_distance_cost
        # version control
        self.version = version
