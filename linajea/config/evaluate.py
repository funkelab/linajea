import attr

from .data import DataConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s
class EvaluateConfig:
    gt_db_name = attr.ib(type=str)
    matching_threshold = attr.ib(type=int)
    data = attr.ib(converter=ensure_cls(DataConfig), default=None)
    job = attr.ib(converter=ensure_cls(JobConfig), default=None)
    from_scratch = attr.ib(type=bool, default=False)
