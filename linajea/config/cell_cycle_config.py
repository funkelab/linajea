import attr

from linajea import load_config
from .cnn_config import (EfficientNetConfig,
                         ResNetConfig,
                         VGGConfig)
from .evaluate import EvaluateCellCycleConfig
from .general import GeneralConfig
from .optimizer import OptimizerConfig
from .predict import PredictCellCycleConfig
from .test import TestCellCycleConfig
from .train import TrainCellCycleConfig
from .utils import ensure_cls
from .validate import ValidateCellCycleConfig


def model_converter():
    def converter(val):
        if val['network_type'].lower() == "vgg":
            return VGGConfig(**val) # type: ignore
        elif val['network_type'].lower() == "resnet":
            return ResNetConfig(**val) # type: ignore
        elif val['network_type'].lower() == "efficientnet":
            return EfficientNetConfig(**val) # type: ignore
        else:
            raise RuntimeError("invalid network_type: {}!".format(
                val['network_type']))
    return converter


@attr.s(kw_only=True)
class CellCycleConfig:
    path = attr.ib(type=str)
    general = attr.ib(converter=ensure_cls(GeneralConfig))
    model = attr.ib(converter=model_converter())
    optimizer = attr.ib(converter=ensure_cls(OptimizerConfig))
    train = attr.ib(converter=ensure_cls(TrainCellCycleConfig))
    test = attr.ib(converter=ensure_cls(TestCellCycleConfig))
    validate = attr.ib(converter=ensure_cls(ValidateCellCycleConfig))
    predict = attr.ib(converter=ensure_cls(PredictCellCycleConfig))
    evaluate = attr.ib(converter=ensure_cls(EvaluateCellCycleConfig))

    @classmethod
    def from_file(cls, path):
        config_dict = load_config(path)
        return cls(path=path, **config_dict) # type: ignore
