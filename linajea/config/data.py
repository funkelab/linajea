import attr
from typing import List

from .utils import ensure_cls


@attr.s(kw_only=True)
class DataConfig:
    filename = attr.ib(type=str)
    group = attr.ib(type=str, default=None)
    voxel_size = attr.ib(type=List[int], default=None)
    roi_offset = attr.ib(type=List[int], default=None)
    roi_shape = attr.ib(type=List[int], default=None)


@attr.s(kw_only=True)
class DataDBConfig:
    @attr.s(kw_only=True)
    class DataDBMetaGeneralConfig:
        setup_dir = attr.ib(type=str)

    @attr.s(kw_only=True)
    class DataDBMetaPredictionConfig:
        iteration = attr.ib(type=int)
        cell_score_threshold = attr.ib(type=float)

    db_name = attr.ib(type=str, default=None)
    general = attr.ib(converter=ensure_cls(DataDBMetaGeneralConfig), default=None)
    prediction = attr.ib(converter=ensure_cls(DataDBMetaPredictionConfig), default=None)
