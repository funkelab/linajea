"""Configuration used to define prediction parameters
"""
import attr

from .augment import NormalizeConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class _PredictConfig:
    """Defines base class for general prediction parameters

    Attributes
    ----------
    job: JobConfig
        HPC cluster parameters, default constructed (executed locally)
        if not supplied
    use_swa: bool
        If also used during training, should the Stochastic Weight Averaging
        model be used for prediction, check TrainConfig for more information
    normalization: NormalizeConfig
        How input data should be normalized, if not set
        train.normalization is used
    """
    job = attr.ib(converter=ensure_cls(JobConfig),
                  default=attr.Factory(JobConfig))
    use_swa = attr.ib(type=bool, default=None)
    normalization = attr.ib(converter=ensure_cls(NormalizeConfig),
                            default=None)


@attr.s(kw_only=True)
class PredictTrackingConfig(_PredictConfig):
    """Defines specialized class for configuration of tracking prediction

    Attributes
    ----------
    write_to_zarr: bool
        Write output to zarr volume, e.g. for visualization
    write_to_db: bool
        Write output to database, required for next steps
    write_db_from_zarr: bool
        Use previously computed zarr to write to database
    no_db_access: bool
        If write_to_zarr is used, do not access database
        (otherwise used to store which blocks have already been predicted)
    processes_per_worker: int
        How many processes each worker can use (for parallel data loading)
    output_zarr_dir: str
        Where zarr should be stored
    """
    write_to_zarr = attr.ib(type=bool, default=False)
    write_to_db = attr.ib(type=bool, default=True)
    write_db_from_zarr = attr.ib(type=bool, default=False)
    no_db_access = attr.ib(type=bool, default=False)
    processes_per_worker = attr.ib(type=int, default=1)
    output_zarr_dir = attr.ib(type=str, default=".")

    def __attrs_post_init__(self):
        """verify that combination of supplied parameters is valid"""
        assert self.write_to_zarr or self.write_to_db, \
            "prediction not written, set write_to_zarr or write_to_db to true!"
        assert not ((self.write_to_db or self.write_db_from_zarr) and
                    self.no_db_access), (
                "no_db_access can only be set if no data is written "
                "to db (it then disables db done checks for write to zarr)")
