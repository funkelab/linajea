import attr
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class PredictConfig:
    job = attr.ib(converter=ensure_cls(JobConfig))
    path_to_script = attr.ib(type=str)
    use_swa = attr.ib(type=bool, default=None)

@attr.s(kw_only=True)
class PredictTrackingConfig(PredictConfig):
    path_to_script_db_from_zarr = attr.ib(type=str, default=None)
    write_to_zarr = attr.ib(type=bool, default=False)
    write_to_db = attr.ib(type=bool, default=True)
    write_db_from_zarr = attr.ib(type=bool, default=False)
    no_db_access = attr.ib(type=bool, default=False)
    processes_per_worker = attr.ib(type=int, default=1)
    output_zarr_prefix = attr.ib(type=str, default=".")

    def __attrs_post_init__(self):
        assert self.write_to_zarr or self.write_to_db, \
            "prediction not written, set write_to_zarr or write_to_db to true!"
        assert not self.write_db_from_zarr or \
            self.path_to_script_db_from_zarr, \
            "supply path_to_script_db_from_zarr if write_db_from_zarr is used!"
        assert not ((self.write_to_db or self.write_db_from_zarr) and self.no_db_access), \
            "no_db_access can only be set if no data is written to db (it then disables db done checks for write to zarr)"

@attr.s(kw_only=True)
class PredictCellCycleConfig(PredictConfig):
    batch_size = attr.ib(type=int)
    max_samples = attr.ib(type=int, default=None)
    prefix = attr.ib(type=str, default="")
