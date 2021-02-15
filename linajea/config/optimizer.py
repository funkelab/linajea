import attr
from typing import List

from .utils import ensure_cls


@attr.s(kw_only=True)
class OptimizerArgsConfig:
    learning_rate = attr.ib(type=float)
    momentum = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class OptimizerKwargsConfig:
    beta1 = attr.ib(type=float, default=None)
    beta2 = attr.ib(type=float, default=None)
    epsilon = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class OptimizerConfig:
    optimizer = attr.ib(type=str, default="AdamOptimizer")
    args = attr.ib(converter=ensure_cls(OptimizerArgsConfig))
    kwargs = attr.ib(converter=ensure_cls(OptimizerKwargsConfig))
