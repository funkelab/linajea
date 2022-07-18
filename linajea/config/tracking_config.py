"""Main configuration used for a tracking experiment

Aggregates the sub configuration modules into a large configuration
file. Most parameters are optional, depending on which step of the
pipeline should be computed. If parameters necessary for a particular
step are not present, an error will be thrown.
"""
import attr
import pymongo

from .data import DataROIConfig
from .evaluate import EvaluateTrackingConfig
from .extract import ExtractConfig
from .general import GeneralConfig
from .optimizer import OptimizerTorchConfig
from .predict import PredictTrackingConfig
from .solve import SolveConfig
from .train_test_validate_data import (InferenceDataTrackingConfig,
                                       TestDataTrackingConfig,
                                       TrainDataTrackingConfig,
                                       ValidateDataTrackingConfig)
from .train import TrainTrackingConfig
from .unet_config import UnetConfig
from .utils import (ensure_cls,
                    load_config)


@attr.s(kw_only=True)
class TrackingConfig:
    """Defines the configuration for a tracking experiment

    Attributes
    ----------
    path: str
        Path to the config file, do not set, will be overwritten and
        set automatically
    general: GeneralConfig
        General configuration parameters
    model: UnetConfig
        Parameters defining the use U-Net
    optimizerTorch: OptimizerTorchConfig
        Parameters defining the optimizer used during training
    train: TrainTrackingConfig
        Parameters defining the general training parameters, e.g.
        augmentation, how many steps
    train_data: TrainDataTrackingConfig
    test_data: TestDataTrackingConfig
    validate_data: ValidateDataTrackingConfig
    inference_data: InferenceDataTrackingConfig
        Which data set to use for training/validation/test/inference
    predict: PredictTrackingConfig
        Prediction parameters
    extract: ExtractConfig
        Parameters defining how edges should be extracted from Candidates
    solve: SolveConfig
        ILP weights and other parameters defining solving step
    evaluate: EvaluateTrackingConfig
        Evaluation parameters
    """
    path = attr.ib(type=str)
    general = attr.ib(converter=ensure_cls(GeneralConfig))
    model = attr.ib(converter=ensure_cls(UnetConfig), default=None)
    optimizerTorch = attr.ib(converter=ensure_cls(OptimizerTorchConfig),
                             default=None)
    train = attr.ib(converter=ensure_cls(TrainTrackingConfig), default=None)
    train_data = attr.ib(converter=ensure_cls(TrainDataTrackingConfig),
                         default=None)
    test_data = attr.ib(converter=ensure_cls(TestDataTrackingConfig),
                        default=None)
    validate_data = attr.ib(converter=ensure_cls(ValidateDataTrackingConfig),
                            default=None)
    inference_data = attr.ib(converter=ensure_cls(InferenceDataTrackingConfig),
                             default=None)
    predict = attr.ib(converter=ensure_cls(PredictTrackingConfig),
                      default=None)
    extract = attr.ib(converter=ensure_cls(ExtractConfig), default=None)
    solve = attr.ib(converter=ensure_cls(SolveConfig), default=None)
    evaluate = attr.ib(converter=ensure_cls(EvaluateTrackingConfig),
                       default=None)

    @classmethod
    def from_file(cls, path):
        """Construct TrackingConfig object from path to config file

        Called as TrackingConfig.from_file(path_to_config_file)
        """
        config_dict = load_config(path)
        config_dict["path"] = path
        return cls(**config_dict)  # type: ignore

    def __attrs_post_init__(self):
        """Validate supplied parameters

        At most one optimizer configuration can be supplied
        If use_swa is not set for prediction step, use value from train
        If normalization is not set for prediction step, use value from train
        Verify that ROI is set at some level for each data source
        """
        if self.predict is not None and \
           self.train is not None and \
           self.predict.use_swa is None:
            self.predict.use_swa = self.train.use_swa

        dss = []
        dss += self.train_data.data_sources \
            if self.train_data is not None else []
        dss += self.test_data.data_sources \
            if self.test_data is not None else []
        dss += self.validate_data.data_sources \
            if self.validate_data is not None else []
        dss += [self.inference_data.data_source] \
            if self.inference_data is not None else []
        for sample in dss:
            if sample.roi is None:
                try:
                    client = pymongo.MongoClient(host=self.general.db_host)
                    db = client[sample.db_name]
                    query_result = db["db_meta_info"].find_one()
                    sample.roi = DataROIConfig(**query_result["roi"])
                except Exception as e:
                    raise RuntimeError(
                        "please specify roi for data! not set and unable to "
                        "determine it automatically based on given db "
                        "(db_name) (%s)" % e)

        if self.predict is not None:
            if self.predict.normalization is None and \
               self.train is not None:
                self.predict.normalization = self.train.normalization
