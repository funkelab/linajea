"""Configuration used to define the optimizer used during training

Which options are available depends on the framework used,
currently only pytorch
"""
from typing import List

import attr

from .utils import ensure_cls


@attr.s(kw_only=True)
class OptimizerTorchKwargsConfig:
    """Defines the keyword (kwargs) arguments for a pytorch optimizer

    Attributes
    ----------
    betas: list of float
        Beta parameters of Adam optimizer
    eps: float
        Epsilon parameter of Adam optimizer
    lr: float
        Which learning rate to use, fixed value
    amsgrad: bool
        Should the amsgrad extension to Adam be used?
    momentum: float
        Which momentum to use, fixed value
    nesterov: bool
        Should Nesterov momentum be used?
    weight_decay: float
        Rate of l2 weight decay

    Notes
    -----
    Extend list of attributes for other optimizers
    """
    betas = attr.ib(type=List[float], default=None)
    eps = attr.ib(type=float, default=None)
    lr = attr.ib(type=float, default=None)
    amsgrad = attr.ib(type=bool, default=None)
    momentum = attr.ib(type=float, default=None)
    nesterov = attr.ib(type=bool, default=None)
    weight_decay = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class OptimizerTorchConfig:
    """Defines which pytorch optimizer to use

    Attributes
    ----------
    optimizer: str
        Name of torch optimizer class to use
    kwargs: OptimizerTorchKwargsConfig
        Which kwargs should be passed to optimizer constructor

    Notes
    -----
    Example on how to create optimzier:
    opt = getattr(torch.optim, config.optimizerTorch.optimizer)(
        model.parameters(), **config.optimizerTorch.get_kwargs())
    """
    optimizer = attr.ib(type=str, default="Adam")
    kwargs = attr.ib(converter=ensure_cls(OptimizerTorchKwargsConfig),
                     default=attr.Factory(OptimizerTorchKwargsConfig))

    def get_kwargs(self):
        """Get dict of keyword parameters"""
        return {a: v for a, v in attr.asdict(self.kwargs).items()
                if v is not None}
