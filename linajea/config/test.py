import attr
from typing import Dict, List

from .data import (DataConfig,
                   DataDBConfig)
from .utils import ensure_cls


@attr.s(kw_only=True)
class TestConfig:
    data = attr.ib(converter=ensure_cls(DataConfig))
    checkpoint = attr.ib(int)
    cell_score_threshold = attr.ib(float)


@attr.s(kw_only=True)
class TestCellCycleConfig (TestConfig):
    use_database = attr.ib(type=bool)
    database = attr.ib(converter=ensure_cls(DataDBConfig), default=None)
