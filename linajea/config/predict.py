import attr
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class PredictConfig:
    job = attr.ib(converter=ensure_cls(JobConfig))


@attr.s(kw_only=True)
class PredictTrackingConfig(PredictConfig):
    path_to_script = attr.ib(type=str)
    write_to_zarr = attr.ib(type=bool, default=False)
    write_to_db = attr.ib(type=bool, default=True)
    processes_per_worker = attr.ib(type=int, default=1)
    output_zarr_prefix = attr.ib(type=str, default=".")


@attr.s(kw_only=True)
class PredictCellCycleConfig(PredictConfig):
    batch_size = attr.ib(type=int)
