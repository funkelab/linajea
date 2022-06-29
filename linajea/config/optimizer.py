"""Configuration used to define the optimizer used during training

Which options are available depends on the framework used,
usually pytorch or tensorflow
"""
from typing import List

import attr

from .utils import ensure_cls


@attr.s(kw_only=True)
class OptimizerTF1ArgsConfig:
    """Defines the positional (args) arguments for a tf1 optimizer

    Attributes
    ----------
    weight_decay: float
        Rate of l2 weight decay
    learning_rate: float
        Which learning rate to use, fixed value
    momentum: float
        Which momentum to use, fixed value
    """
    weight_decay = attr.ib(type=float, default=None)
    learning_rate = attr.ib(type=float, default=None)
    momentum = attr.ib(type=float, default=None)

@attr.s(kw_only=True)
class OptimizerTF1KwargsConfig:
    """Defines the keyword (kwargs) arguments for a tf1 optimizer

    If not set, framework defaults are used.

    Attributes
    ----------
    learning_rate: float
        Which learning rate to use, fixed value
    beta1: float
        Beta1 parameter of Adam optimizer
    beta2: float
        Beta2 parameter of Adam optimzer
    epsilon: float
        Epsilon parameter of Adam optimizer

    Notes
    -----
    Extend list of attributes for other optimizers
    """
    learning_rate = attr.ib(type=float, default=None)
    beta1 = attr.ib(type=float, default=None)
    beta2 = attr.ib(type=float, default=None)
    epsilon = attr.ib(type=float, default=None)

@attr.s(kw_only=True)
class OptimizerTF1Config:
    """Defines which tensorflow1 optimizer to use

    Attributes
    ----------
    optimizer: str
        Name of tf1 optimizer class to use
    lr_schedule: str
        What type of learning rate schedule to use, deprecated
    args: OptimizerTF1ArgsConfig
        Which args should be passed to optimizer constructor
    kwargs: OptimizerTF1KwargsConfig
        Which kwargs should be passed to optimizer constructor

    Notes
    -----
    Example on how to create optimzier:
    opt = getattr(tf.train, config.optimizerTF1.optimizer)(
            *config.optimizerTF1.get_args(),
            **config.optimizerTF1.get_kwargs())
    """
    optimizer = attr.ib(type=str, default="AdamOptimizer")
    lr_schedule = attr.ib(type=str, default=None)
    args = attr.ib(converter=ensure_cls(OptimizerTF1ArgsConfig))
    kwargs = attr.ib(converter=ensure_cls(OptimizerTF1KwargsConfig))

    def get_args(self):
        """Get list of positional parameters"""
        return [v for v in attr.astuple(self.args) if v is not None]

    def get_kwargs(self):
        """Get dict of keyword parameters"""
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}


@attr.s(kw_only=True)
class OptimizerTF2KwargsConfig:
    """Defines the keyword (kwargs) arguments for a tf2 optimizer

    If not set, framework defaults are used.

    Attributes
    ----------
    beta_1: float
        Beta1 parameter of Adam optimizer
    beta_2: float
        Beta2 parameter of Adam optimzer
    epsilon: float
        Epsilon parameter of Adam optimizer
    learning_rate: float
        Which learning rate to use, fixed value
    momentum: float
        Which momentum to use, fixed value

    Notes
    -----
    Extend list of attributes for other optimizers
    """
    beta_1 = attr.ib(type=float, default=None)
    beta_2 = attr.ib(type=float, default=None)
    epsilon = attr.ib(type=float, default=None)
    learning_rate = attr.ib(type=float, default=None)
    momentum = attr.ib(type=float, default=None)

@attr.s(kw_only=True)
class OptimizerTF2Config:
    """Defines which tensorflow1 optimizer to use

    Attributes
    ----------
    optimizer: str
        Name of tf2 optimizer class to use
    kwargs: OptimizerTF2KwargsConfig
        Which kwargs should be passed to optimizer constructor

    Notes
    -----
    Example on how to create optimzier:
    opt = getattr(tf.keras.optimizers, config.optimizerTF2.optimizer)(
        **config.optimizerTF2.get_kwargs())
    """
    optimizer = attr.ib(type=str, default="Adam")
    kwargs = attr.ib(converter=ensure_cls(OptimizerTF2KwargsConfig))

    def get_kwargs(self):
        """Get dict of keyword parameters"""
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}


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
    kwargs = attr.ib(converter=ensure_cls(OptimizerTorchKwargsConfig))

    def get_kwargs(self):
        """Get dict of keyword parameters"""
        return {a:v for a,v in attr.asdict(self.kwargs).items() if v is not None}
