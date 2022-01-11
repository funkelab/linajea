import attr
from typing import List

from .utils import ensure_cls


@attr.s(kw_only=True)
class AugmentElasticConfig:
    control_point_spacing = attr.ib(type=List[int])
    jitter_sigma = attr.ib(type=List[int])
    rotation_min = attr.ib(type=int)
    rotation_max = attr.ib(type=int)
    rotation_3d = attr.ib(type=bool, default=False)
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
class AugmentNoiseSpeckleConfig:
    var = attr.ib(type=float)

@attr.s(kw_only=True)
class AugmentNoiseSaltPepperConfig:
    amount = attr.ib(type=float)


@attr.s(kw_only=True)
class AugmentJitterConfig:
    jitter = attr.ib(type=List[int])

@attr.s(kw_only=True)
class AugmentZoomConfig:
    factor_min = attr.ib(type=float)
    factor_max = attr.ib(type=float)
    spatial_dims = attr.ib(type=int)

@attr.s(kw_only=True)
class AugmentHistogramConfig:
    range_low = attr.ib(type=float)
    range_high = attr.ib(type=float)
    after_int_aug = attr.ib(type=bool, default=True)


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
    noise_speckle = attr.ib(converter=ensure_cls(AugmentNoiseSpeckleConfig),
                             default=None)
    noise_saltpepper = attr.ib(converter=ensure_cls(AugmentNoiseSaltPepperConfig),
                               default=None)
    zoom = attr.ib(converter=ensure_cls(AugmentZoomConfig),
                   default=None)
    histogram = attr.ib(converter=ensure_cls(AugmentHistogramConfig),
                        default=None)


@attr.s(kw_only=True)
class AugmentTrackingConfig(AugmentConfig):
    reject_empty_prob = attr.ib(type=float) # (default=1.0?)
    norm_bounds = attr.ib(type=List[int], default=None)
    divisions = attr.ib(type=bool) # float for percentage?
    normalization = attr.ib(type=str, default=None)
    perc_min = attr.ib(type=str, default=None)
    perc_max = attr.ib(type=str, default=None)
    point_balance_radius = attr.ib(type=int, default=1)


@attr.s(kw_only=True)
class AugmentCellCycleConfig(AugmentConfig):
    min_key = attr.ib(type=str, default=None)
    max_key = attr.ib(type=str, default=None)
    norm_min = attr.ib(type=int, default=None)
    norm_max = attr.ib(type=int, default=None)
    jitter = attr.ib(converter=ensure_cls(AugmentJitterConfig),
                     default=None)
