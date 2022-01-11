import attr

from linajea import load_config
from .evaluate import EvaluateTrackingConfig
from .extract import ExtractConfig
from .general import GeneralConfig
from .optimizer import (OptimizerTF1Config,
                        OptimizerTF2Config,
                        OptimizerTorchConfig)
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
    optimizerTF1 = attr.ib(converter=ensure_cls(OptimizerTF1Config), default=None)
    optimizerTF2 = attr.ib(converter=ensure_cls(OptimizerTF2Config), default=None)
    optimizerTorch = attr.ib(converter=ensure_cls(OptimizerTorchConfig), default=None)
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

    def __attrs_post_init__(self):
        assert (int(bool(self.optimizerTF1)) +
                int(bool(self.optimizerTF2)) +
                int(bool(self.optimizerTorch))) == 1, \
        "please specify exactly one optimizer config (tf1, tf2, torch)"

        if self.predict.use_swa is None:
            self.predict.use_swa = self.train.use_swa
