import attr
from typing import List, Dict

from .job import JobConfig
from .utils import ensure_cls


def edge_move_converter():
    def converter(val):
        if isinstance(val, int):
            return {0: val}
        else:
            return val
    return converter


@attr.s
class ExtractConfig:
    edge_move_threshold = attr.ib(type=Dict[int, int],
                                  converter=edge_move_converter())
    block_size = attr.ib(type=List[int])
    job = attr.ib(converter=ensure_cls(JobConfig))
    context = attr.ib(type=List[int], default=None)
