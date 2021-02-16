import attr
from typing import List

from .utils import ensure_cls


@attr.s(kw_only=True)
class DataFileConfig:
    filename = attr.ib(type=str)
    group = attr.ib(type=str, default=None)
    voxel_size = attr.ib(type=List[int], default=None)


@attr.s(kw_only=True)
class DataDBConfig:
    db_name = attr.ib(type=str, default=None)
    setup_dir = attr.ib(type=str, default=None)
    checkpoint = attr.ib(type=int)
    cell_score_threshold = attr.ib(type=float)


@attr.s(kw_only=True)
class DataROIConfig:
    offset = attr.ib(type=List[int], default=None)
    shape = attr.ib(type=List[int], default=None)


@attr.s(kw_only=True)
class DataConfig:
    datafile = attr.ib(converter=ensure_cls(DataFileConfig))
    database = attr.ib(converter=ensure_cls(DataDBConfig), default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig))
