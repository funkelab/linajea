import attr

from .data import DataROIConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class EvaluateConfig:
    gt_db_name = attr.ib(type=str)
    matching_threshold = attr.ib(type=int)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
    job = attr.ib(converter=ensure_cls(JobConfig), default=None)
    from_scratch = attr.ib(type=bool, default=False)


@attr.s(kw_only=True)
class EvaluateTrackingConfig(EvaluateConfig):
    from_scratch = attr.ib(type=bool, default=False)


@attr.s(kw_only=True)
class EvaluateCellCycleConfig(EvaluateConfig):
    max_samples = attr.ib(type=int)
    metric = attr.ib(type=str)
    use_database = attr.ib(type=bool, default=True)
    one_off = attr.ib(type=bool)
    prob_threshold = attr.ib(type=float)
    dry_run = attr.ib(type=bool)
    find_fn = attr.ib(type=bool)
