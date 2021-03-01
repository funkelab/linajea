import attr
from typing import Dict, List

from .augment import (AugmentTrackingConfig,
                      AugmentCellCycleConfig)
from .train_test_validate_data import TrainDataTrackingConfig
from .job import JobConfig
from .utils import ensure_cls


def use_radius_converter():
    def converter(val):
        if isinstance(val, bool):
            return {0: val}
        else:
            return val
    return converter


@attr.s(kw_only=True)
class TrainConfig:
    path_to_script = attr.ib(type=str)
    job = attr.ib(converter=ensure_cls(JobConfig))
    cache_size = attr.ib(type=int)
    max_iterations = attr.ib(type=int)
    checkpoint_stride = attr.ib(type=int)
    snapshot_stride = attr.ib(type=int)
    profiling_stride = attr.ib(type=int)
    use_tf_data = attr.ib(type=bool, default=False)
    use_auto_mixed_precision = attr.ib(type=bool, default=False)
    val_log_step = attr.ib(type=int)


@attr.s(kw_only=True)
class TrainTrackingConfig(TrainConfig):
    # (radius for binary map -> *2) (optional)
    parent_radius = attr.ib(type=List[float])
    # (sigma for Gauss -> ~*4 (5 in z -> in 3 slices)) (optional)
    rasterize_radius = attr.ib(type=List[float])
    augment = attr.ib(converter=ensure_cls(AugmentTrackingConfig))
    parent_vectors_loss_transition = attr.ib(type=int, default=50000)
    use_radius = attr.ib(type=Dict[int, int],
                         converter=use_radius_converter())


@attr.s(kw_only=True)
class TrainCellCycleConfig(TrainConfig):
    batch_size = attr.ib(type=int)
    augment = attr.ib(converter=ensure_cls(AugmentCellCycleConfig))
    # use_database = attr.ib(type=bool)
    # database = attr.ib(converter=ensure_cls(DataDBConfig), default=None)
