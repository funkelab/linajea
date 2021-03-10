import attr
from typing import List

from .utils import ensure_cls


@attr.s(kw_only=True)
class AugmentElasticConfig:
    control_point_spacing = attr.ib(type=List[int])
    jitter_sigma = attr.ib(type=List[int])
    rotation_min = attr.ib(type=int)
    rotation_max = attr.ib(type=int)
    subsample = attr.ib(type=int, default=1)
    use_fast_points_transform = attr.ib(type=bool, default=False)


@attr.s(kw_only=True)
class AugmentIntensityConfig:
    scale = attr.ib(type=List[float])
    shift = attr.ib(type=List[float])


@attr.s(kw_only=True)
class AugmentShiftConfig:
    prob_slip = attr.ib(type=float)
    prob_shift = attr.ib(type=float)
    sigma = attr.ib(type=List[int])


@attr.s(kw_only=True)
class AugmentSimpleConfig:
    mirror = attr.ib(type=List[int])
    transpose = attr.ib(type=List[int])


@attr.s(kw_only=True)
class AugmentNoiseGaussianConfig:
    var = attr.ib(type=float)


@attr.s(kw_only=True)
class AugmentNoiseSaltPepperConfig:
    amount = attr.ib(type=float)


@attr.s(kw_only=True)
class AugmentJitterConfig:
    jitter = attr.ib(type=List[int])


@attr.s(kw_only=True)
class AugmentConfig:
    elastic = attr.ib(converter=ensure_cls(AugmentElasticConfig),
                      default=None)
    shift = attr.ib(converter=ensure_cls(AugmentShiftConfig),
                    default=None)
    intensity = attr.ib(converter=ensure_cls(AugmentIntensityConfig),
                        default=None)
    simple = attr.ib(converter=ensure_cls(AugmentSimpleConfig),
                     default=None)
    noise_gaussian = attr.ib(converter=ensure_cls(AugmentNoiseGaussianConfig),
                             default=None)
    noise_saltpepper = attr.ib(converter=ensure_cls(AugmentNoiseSaltPepperConfig),
                               default=None)


@attr.s(kw_only=True)
class AugmentTrackingConfig(AugmentConfig):
    reject_empty_prob = attr.ib(type=float) # (default=1.0?)
    norm_bounds = attr.ib(type=List[int])
    divisions = attr.ib(type=bool) # float for percentage?
    normalization = attr.ib(type=str, default=None)
    perc_min = attr.ib(type=str)
    perc_max = attr.ib(type=str)


@attr.s(kw_only=True)
class AugmentCellCycleConfig(AugmentConfig):
    min_key = attr.ib(type=str)
    max_key = attr.ib(type=str)
    norm_min = attr.ib(type=int, default=None)
    norm_max = attr.ib(type=int, default=None)
    jitter = attr.ib(converter=ensure_cls(AugmentJitterConfig),
                     default=None)
