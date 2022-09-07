"""Configuration used to define model training parameter
"""
from typing import Dict, List

import attr

from .augment import (AugmentTrackingConfig,
                      NormalizeConfig)
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class _TrainConfig:
    """Defines base class for general training parameters

    Attributes
    ----------
    job: JobConfig
        HPC cluster parameters
    cache_size: int
        How many batches to cache/precompute in parallel
    max_iterations: int
        For how many steps to train
    checkpoint_stride, snapshot_stride, profiling_stride: int
        After how many steps a checkpoint (of trained model) or
        snapshot (of input and output data for one step) should be
        stored or profling information printed
    use_auto_mixed_precision: bool
        Use automatic mixed precision, reduces memory consumption,
        should not impact final performance
    use_swa: bool
        Use Stochastic Weight Averaging, in some preliminary tests we
        observed improvements for the tracking networks but not for
        others. Cheap to compute. Keeps two copies of
        the model weights in memory, one is updated as usual, the other
        is a weighted average of the main weights every x steps.
        Has to be supported by the respective gunpowder train node.
    swa_every_it: bool
        Compute SWA every step (after swa_start_it)
    swa_start_it: int
        Start computing SWA after this step, network should be converged
    swa_freq_it: int
        Compute SWA every n-th step, after swa_start_it
    val_log_step: int
        Interleave a validation step every n-th training step,
        has to be supported by the training script
    normalization: NormalizeConfig
        Parameters defining which data normalization type to use
    """
    job = attr.ib(converter=ensure_cls(JobConfig),
                  default=attr.Factory(JobConfig))
    cache_size = attr.ib(type=int, default=8)
    max_iterations = attr.ib(type=int)
    checkpoint_stride = attr.ib(type=int)
    snapshot_stride = attr.ib(type=int, default=1000)
    profiling_stride = attr.ib(type=int, default=500)
    use_auto_mixed_precision = attr.ib(type=bool, default=False)
    use_swa = attr.ib(type=bool, default=False)
    swa_every_it = attr.ib(type=bool, default=False)
    swa_start_it = attr.ib(type=int, default=None)
    swa_freq_it = attr.ib(type=int, default=None)
    val_log_step = attr.ib(type=int, default=None)
    normalization = attr.ib(converter=ensure_cls(NormalizeConfig),
                            default=None)

    def __attrs_post_init__(self):
        if self.use_swa:
            assert (self.swa_start_it is not None and
                    self.swa_freq_it is not None), \
                "if swa is used, please set start and freq it"


@attr.s(kw_only=True)
class TrainTrackingConfig(_TrainConfig):
    """Defines specialized class for configuration of tracking training

    Attributes
    ----------
    object_radius: list of float
        Radius around each object in which GT movement vectors
        are drawn, in world units, one value per dimension
        (radius for binary map -> *2 to get diameter,
        optional if annotations contain radius/use_radius = True)
    move_radius: float
        Extend ROI by this much context to ensure all parent objects are
        inside the ROI, in world units
    rasterize_radius: list of float
        Standard deviation of Gauss kernel used to draw Gaussian blobs
        for object indicator network, one value per dimension
        (*4 to get approx. width of blob, for 5x anisotropic data set
        value in z to 5 for it to be in 3 slices,
        optional if annotations contain radius/use_radius = True)
    augment: AugmentTrackingConfig
        Defines data augmentations used during training,
        see AugmentTrackingConfig
    movement_vectors_loss_transition_factor: float
        We transition from computing the movement vector loss for
        all pixels smoothly to computing it only on the peak pixels of
        the object indicator map, this value determines the speed of the
        transition
    movement_vectors_loss_transition_offset: int
        This value marks the training step when the transition blending
        factor is 0.5
    use_radius: dict of int: int
        If the GT annotations contain radii, enable this to use them
        instead of the fixed values defined above
    """
    object_radius = attr.ib(type=List[float])
    move_radius = attr.ib(type=float)
    rasterize_radius = attr.ib(type=List[float])
    augment = attr.ib(converter=ensure_cls(AugmentTrackingConfig),
                      default=attr.Factory(AugmentTrackingConfig))
    movement_vectors_loss_transition_factor = attr.ib(type=float, default=0.01)
    movement_vectors_loss_transition_offset = attr.ib(type=int, default=20000)
    use_radius = attr.ib(type=Dict[int, int], default=None)
