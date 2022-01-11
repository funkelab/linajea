import attr
from typing import List

from .utils import ensure_cls


@attr.s(kw_only=True)
class OptimizerTF1ArgsConfig:
    weight_decay = attr.ib(type=float, default=None)
    learning_rate = attr.ib(type=float, default=None)
    momentum = attr.ib(type=float, default=None)

@attr.s(kw_only=True)
class OptimizerTF1KwargsConfig:
    learning_rate = attr.ib(type=float, default=None)
    beta1 = attr.ib(type=float, default=None)
    beta2 = attr.ib(type=float, default=None)
    epsilon = attr.ib(type=float, default=None)
    # to be extended for other (not Adam) optimizers

@attr.s(kw_only=True)
class OptimizerTF1Config:
    optimizer = attr.ib(type=str, default="AdamOptimizer")
    lr_schedule = attr.ib(type=str, default=None)
    args = attr.ib(converter=ensure_cls(OptimizerTF1ArgsConfig))
    kwargs = attr.ib(converter=ensure_cls(OptimizerTF1KwargsConfig))

    def get_args(self):
        return [v for v in attr.astuple(self.args) if v is not None]

    def get_kwargs(self):
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}


@attr.s(kw_only=True)
class OptimizerTF2KwargsConfig:
    beta_1 = attr.ib(type=float, default=None)
    beta_2 = attr.ib(type=float, default=None)
    epsilon = attr.ib(type=float, default=None)
    learning_rate = attr.ib(type=float, default=None)
    momentum = attr.ib(type=float, default=None)
    # to be extended for other (not Adam) optimizers

@attr.s(kw_only=True)
class OptimizerTF2Config:
    optimizer = attr.ib(type=str, default="Adam")
    kwargs = attr.ib(converter=ensure_cls(OptimizerTF2KwargsConfig))

    def get_kwargs(self):
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}


@attr.s(kw_only=True)
class OptimizerTorchKwargsConfig:
    betas = attr.ib(type=List[float], default=None)
    eps = attr.ib(type=float, default=None)
    lr = attr.ib(type=float, default=None)
    amsgrad = attr.ib(type=bool, default=None)
    momentum = attr.ib(type=float, default=None)
    nesterov = attr.ib(type=bool, default=None)
    weight_decay = attr.ib(type=float, default=None)
    # to be extended for other (not Adam) optimizers

@attr.s(kw_only=True)
class OptimizerTorchConfig:
    optimizer = attr.ib(type=str, default="Adam")
    kwargs = attr.ib(converter=ensure_cls(OptimizerTorchKwargsConfig))

    def get_kwargs(self):
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}
