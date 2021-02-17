import attr
from typing import Dict, List

from .data import (DataFileConfig,
                   DataDBConfig)
from .utils import ensure_cls


@attr.s(kw_only=True)
class ValidateConfig:
    data = attr.ib(converter=ensure_cls(DataFileConfig))
    database = attr.ib(converter=ensure_cls(DataDBConfig))
    checkpoints = attr.ib(type=List[int])
    cell_score_threshold = attr.ib(type=List[float])


@attr.s(kw_only=True)
class ValidateCellCycleConfig (ValidateConfig):
    use_database = attr.ib(type=bool)
    database = attr.ib(converter=ensure_cls(DataDBConfig), default=None)
