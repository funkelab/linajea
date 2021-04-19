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

    def get_args(self):
        return [v for v in attr.astuple(self.args) if v is not None]

    def get_kwargs(self):
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}
