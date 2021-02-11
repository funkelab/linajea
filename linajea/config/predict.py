import attr
from .data import DataConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s
class PredictConfig:
    data = attr.ib(converter=ensure_cls(DataConfig))
    job = attr.ib(converter=ensure_cls(JobConfig))
    iteration = attr.ib(type=int)
    cell_score_threshold = attr.ib(type=float)
    write_to_zarr = attr.ib(type=bool, default=False)
    write_to_db = attr.ib(type=bool, default=True)
    processes_per_worker = attr.ib(type=int, default=1)
