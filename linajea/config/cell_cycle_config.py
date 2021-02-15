import attr

from .cnn_config import (EfficientNetConfig,
                         ResNetConfig,
                         VGGConfig)
from .evaluate import EvaluateConfig
from .general import GeneralConfig
from .optimizer import OptimizerConfig
from .predict import PredictConfig
from .train import TrainCellCycleConfig
from .utils import ensure_cls


def model_converter():
    def converter(val):
        if val['network_type'].lower() == "vgg":
            return VGGConfig(**val)
        elif val['network_type'].lower() == "resnet":
            return ResNetConfig(**val)
        elif val['network_type'].lower() == "efficientnet":
            return EfficientNetConfig(**val)
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
    predict = attr.ib(converter=ensure_cls(PredictConfig))
    evaluate = attr.ib(converter=ensure_cls(EvaluateConfig))

    @classmethod
    def from_file(cls, path):
        config_dict = load_config(path)
        return cls(path, **config_dict)
