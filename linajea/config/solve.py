import attr
from typing import List

from .job import JobConfig
from .utils import (ensure_cls,
                    ensure_cls_list)


@attr.s(kw_only=True)
class SolveParametersConfig:
    cost_appear = attr.ib(type=float)
    cost_disappear = attr.ib(type=float)
    cost_split = attr.ib(type=float)
    threshold_node_score = attr.ib(type=float)
    weight_node_score = attr.ib(type=float)
    threshold_edge_score = attr.ib(type=float)
    weight_prediction_distance_cost = attr.ib(type=float)
    block_size = attr.ib(type=List[int])
    context = attr.ib(type=List[int])
    # max_cell_move: currently use edge_move_threshold from extract
    max_cell_move = attr.ib(type=int, default=None)


@attr.s(kw_only=True)
class SolveConfig:
    job = attr.ib(converter=ensure_cls(JobConfig))
    from_scratch = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=ensure_cls_list(SolveParametersConfig))

    def __attrs_post_init__(self):
        # block size and context must be the same for all parameters!
        block_size = self.parameters[0].block_size
        context = self.parameters[0].context
        for i in range(len(self.parameters)):
            assert block_size == self.parameters[i].block_size, \
                "%s not equal to %s" %\
                (block_size, self.parameters[i].block_size)
            assert context == self.parameters[i].context, \
                "%s not equal to %s" %\
                (context, self.parameters[i].context)
