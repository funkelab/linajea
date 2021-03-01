import attr

from linajea import load_config
from .evaluate import EvaluateTrackingConfig
from .extract import ExtractConfig
from .general import GeneralConfig
from .optimizer import OptimizerConfig
from .predict import PredictTrackingConfig
from .solve import SolveConfig
# from .test import TestTrackingConfig
from .train_test_validate_data import (InferenceDataTrackingConfig,
                                       TestDataTrackingConfig,
                                       TrainDataTrackingConfig,
                                       ValidateDataTrackingConfig)
from .train import TrainTrackingConfig
from .unet_config import UnetConfig
from .utils import ensure_cls
# from .validate import ValidateConfig

@attr.s(kw_only=True)
class TrackingConfig:
    path = attr.ib(type=str)
    general = attr.ib(converter=ensure_cls(GeneralConfig))
    model = attr.ib(converter=ensure_cls(UnetConfig))
    optimizer = attr.ib(converter=ensure_cls(OptimizerConfig))
    train = attr.ib(converter=ensure_cls(TrainTrackingConfig))
    train_data = attr.ib(converter=ensure_cls(TrainDataTrackingConfig))
    test_data = attr.ib(converter=ensure_cls(TestDataTrackingConfig))
    validate_data = attr.ib(converter=ensure_cls(ValidateDataTrackingConfig))
    inference = attr.ib(converter=ensure_cls(InferenceDataTrackingConfig), default=None)
    predict = attr.ib(converter=ensure_cls(PredictTrackingConfig))
    extract = attr.ib(converter=ensure_cls(ExtractConfig))
    solve = attr.ib(converter=ensure_cls(SolveConfig))
    evaluate = attr.ib(converter=ensure_cls(EvaluateTrackingConfig))

    @classmethod
    def from_file(cls, path):
        config_dict = load_config(path)
        config_dict["path"] = path
        return cls(**config_dict) # type: ignore
