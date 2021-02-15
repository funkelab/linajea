import attr

from linajea import load_config

from .evaluate import EvaluateConfig
from .extract import ExtractConfig
from .general import GeneralConfig
from .optimizer import OptimizerConfig
from .predict import PredictConfig
from .solve import SolveConfig
from .train import TrainTrackingConfig
from .unet_config import UnetConfig
from .utils import ensure_cls


@attr.s
class TrackingConfig:
    path = attr.ib(type=str)
    general = attr.ib(converter=ensure_cls(GeneralConfig))
    model = attr.ib(converter=ensure_cls(UnetConfig))
    optimizer = attr.ib(converter=ensure_cls(OptimizerConfig))
    train = attr.ib(converter=ensure_cls(TrainTrackingConfig))
    predict = attr.ib(converter=ensure_cls(PredictConfig))
    extract = attr.ib(converter=ensure_cls(ExtractConfig))
    solve = attr.ib(converter=ensure_cls(SolveConfig))
    evaluate = attr.ib(converter=ensure_cls(EvaluateConfig))

    @classmethod
    def from_file(cls, path):
        config_dict = load_config(path)
        return cls(path, **config_dict)
