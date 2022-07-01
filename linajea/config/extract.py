"""Configuration for extract edges step

After all object/node candidates have been predicted, for each node
check neighborhood in previous frame for potential parent nodes and
generate edge candidates. Each candidate is scores by the distance
between the location predicted by the movement vector and its actual
position.
"""
from typing import List, Dict

import attr

from .job import JobConfig
from .utils import ensure_cls


def _edge_move_converter():
    def converter(val):
        if isinstance(val, int):
            return {-1: val}
        else:
            return val
    return converter


@attr.s(kw_only=True)
class ExtractConfig:
    """Defines configuration of extract edges step

    Attributes
    ----------
    edge_move_threshold: dict of int: int or int
        How far can a cell move in one frame? If scalar, same value for
        all frames, otherwise dict of frames to thresholds. For each
        node use entry that is higher but closest.
    block_size: list of int
        Large data samples have to be processed in blocks, defines
        size of each block
    job: JobConfig
        HPC cluster parameters, default constructed (executed locally)
        if not supplied
    context: list of int
        Size of context by which block is grown, to ensure consistent
        solution along borders
    use_mv_distance: bool
        Use distance to location predicted by movement vector to look
        for closest neighbors, recommended
    """
    edge_move_threshold = attr.ib(type=Dict[int, int],
                                  converter=_edge_move_converter())
    block_size = attr.ib(type=List[int])
    job = attr.ib(converter=ensure_cls(JobConfig),
                  default=attr.Factory(JobConfig))
    context = attr.ib(type=List[int], default=None)
    use_mv_distance = attr.ib(type=bool, default=False)
