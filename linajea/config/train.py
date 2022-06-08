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
            return {'0': val}
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
    use_swa = attr.ib(type=bool, default=False)
    swa_every_it = attr.ib(type=bool, default=False)
    swa_start_it = attr.ib(type=int, default=None)
    swa_freq_it = attr.ib(type=int, default=None)
    use_grad_norm = attr.ib(type=bool, default=False)
    val_log_step = attr.ib(type=int, default=None)

    def __attrs_post_init__(self):
        if self.use_swa:
            assert self.swa_start_it is not None and self.swa_freq_it is not None, \
                "if swa is used, please set start and freq it"


@attr.s(kw_only=True)
class TrainTrackingConfig(TrainConfig):
    # (radius for binary map -> *2) (optional)
    parent_radius = attr.ib(type=List[float])
    # context to be able to get location of parents during add_parent_vectors
    move_radius = attr.ib(type=float, default=None)
    # (sigma for Gauss -> ~*4 (5 in z -> in 3 slices)) (optional)
    rasterize_radius = attr.ib(type=List[float])
    augment = attr.ib(converter=ensure_cls(AugmentTrackingConfig))
    parent_vectors_loss_transition_factor = attr.ib(type=float, default=0.01)
    parent_vectors_loss_transition_offset = attr.ib(type=int, default=20000)
    use_radius = attr.ib(type=Dict[int, int], default=None,
                         converter=use_radius_converter())
    cell_density = attr.ib(default=None)


@attr.s(kw_only=True)
class TrainCellCycleConfig(TrainConfig):
    batch_size = attr.ib(type=int)
    augment = attr.ib(converter=ensure_cls(AugmentCellCycleConfig))
