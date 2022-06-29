"""Linajea configuration

This module defines the components of the configuration files used
by linajea.

Submodules
----------

augment:
    defines configuration used for data augmentation during training
data:
    defines configuration used for data samples
evaluate:
    defines configuration used for evaluation of results
extract:
    defines configuration used to extract edges from predicted nodes
general:
    defines general configuration used by several steps
job:
    defines configuration used to create HPC cluster jobs
predict:
    defines configuration used to predict cells and vectors
solve:
    defines configuration used by ILP to compute tracks
tracking_config:
    combines other config submodules into config for tracking model
train:
    defines configuration used for model training
train_test_validate_data:
    defines configuration to assemble samples to train/test/val data
unet_config:
    defines configuration used for U-Net (e.g. for tracking model)
utils:
    defines utility functions to read and handle config
"""
# flake8: noqa

from .augment import AugmentTrackingConfig
from .data import DataFileConfig
from .evaluate import EvaluateTrackingConfig
from .extract import ExtractConfig
from .general import GeneralConfig
from .job import JobConfig
from .predict import PredictTrackingConfig
from .solve import (SolveConfig,
                    SolveParametersConfig)
from .tracking_config import TrackingConfig
from .train import TrainTrackingConfig
from .train_test_validate_data import InferenceDataTrackingConfig
from .unet_config import UnetConfig
from .utils import (dump_config,
                    load_config,
                    maybe_fix_config_paths_to_machine_and_load)
