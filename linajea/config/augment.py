"""Data augmentation configuration

This modules defines the data augmentation configuration options.
Options are:
 - Elastic
 - Intensity
 - Shift
 - Shuffle Input Channels
 - Gaussian Noise
 - Speckle Noise
 - Salt and Pepper Noise
 - Jitter Noise
 - Zoom
 - Histogram

Notes
-----
Only augmentation specified in configuration file is used,
default is off
"""
from typing import List

import attr

from .utils import ensure_cls


@attr.s(kw_only=True)
class AugmentElasticConfig:
    """Defines options for elastic augment

    Attributes
    ----------
    control_point_spacing: list of int
        Distance between adjacent control points, one value per dimension
    jitter_sigma: list of int
        Sigma used to shift control points, one value per dimension
    rotation_min, rotation_max: int
        Range of angles by which data is rotated, in degree
    rotation_3d: bool
        Use 3d rotation or successive 2d rotations
    subsample: int
        Subsample ROI to speed up computation, up to 4 is usually fine
    use_fast_points_transform: bool
        Use fast transform to augment points, not as well tested

    """
    control_point_spacing = attr.ib(type=List[int], default=[15, 15, 15])
    jitter_sigma = attr.ib(type=List[int], default=[1, 1, 1])
    rotation_min = attr.ib(type=int, default=-45)
    rotation_max = attr.ib(type=int, default=45)
    rotation_3d = attr.ib(type=bool, default=False)
    subsample = attr.ib(type=int, default=1)
    use_fast_points_transform = attr.ib(type=bool, default=False)


@attr.s(kw_only=True)
class AugmentIntensityConfig:
    """Defines options for intensity augment

    Attributes
    ----------
    scale: list of float
        Expects two values, lower and upper range of random factor by
        which data is multiplied/scaled, usually slightly smaller and
        larger than 1
    shift: list of float
        Expects two values, lower and upper range of random term
        which is added to data/by which data is shifted, usually
        slightly smaller and larger than 0
    """
    scale = attr.ib(type=List[float], default=[0.9, 1.1])
    shift = attr.ib(type=List[float], default=[-0.001, 0.001])


@attr.s(kw_only=True)
class AugmentShiftConfig:
    """Defines options for shift augment

    Attributes
    ----------
    prob_slip: float
        Probability that center frame is shifted (independently)
    prob_shift: float
        Probability that center frame and all following ones are shifted
    sigma: list of int
        Standard deviation of shift in each dimension, if single value
        extended to all dimensions.
    """
    prob_slip = attr.ib(type=float, default=0.2)
    prob_shift = attr.ib(type=float, default=0.2)
    sigma = attr.ib(type=List[int], default=[0, 4, 4, 4])


@attr.s(kw_only=True)
class AugmentSimpleConfig:
    """Defines options for simple flip and transpose augment

    Attributes
    ----------
    mirror, transpose: list of int
        List of dimensions to be mirrored/transposed,
        e.g. for 3d+t data and ignoring time: [1, 2, 3]

    Notes
    -----
    It might make sense to not transpose z for anisotropic data
    """
    mirror = attr.ib(type=List[int], default=[1, 2, 3])
    transpose = attr.ib(type=List[int], default=[1, 2, 3])


@attr.s(kw_only=True)
class AugmentNoiseGaussianConfig:
    """Defines options for Gaussian noise augment, used scikit-image

    Attributes
    ----------
    var: float
        Variance of Gaussian Noise
    """
    var = attr.ib(type=float, default=0.01)

@attr.s(kw_only=True)
class AugmentNoiseSpeckleConfig:
    """Defines options for Speckle noise augment, uses scikit-image

    Attributes
    ----------
    var: float
        Variance of Speckle Noise
    """
    var = attr.ib(type=float, default=0.05)

@attr.s(kw_only=True)
class AugmentNoiseSaltPepperConfig:
    """Defines options for S&P noise augment, uses scikit-image

    Attributes
    ----------
    amount: float
        Amount of S&P noise to be added
    """
    amount = attr.ib(type=float, default=0.0001)


@attr.s(kw_only=True)
class AugmentJitterConfig:
    """Defines options for Jitter noise augment

    Notes
    -----
    Only used in cell state classifier, not in tracking

    Attributes
    ----------
    jitter: list of int
        How far to shift cell location, one value per dimension,
        sampled uniformly from [-j, +j] in each dimension
    """
    jitter = attr.ib(type=List[int], default=[0, 3, 3, 3])

@attr.s(kw_only=True)
class AugmentZoomConfig:
    """Defines options for Zoom augment

    Attributes
    ----------
    factor_min, factor_max: float
        Range of random factor by which to zoom in/out, usually
        slightly smaller and larger than 1
    spatial_dims: int
        Number of spatial dimensions, typically 2 or 3,
        assumed to be the last ones (i.e., data[-spatial_dims])
    """
    factor_min = attr.ib(type=float, default=0.85)
    factor_max = attr.ib(type=float, default=1.25)
    spatial_dims = attr.ib(type=int, default=3)

@attr.s(kw_only=True)
class AugmentHistogramConfig:
    """Defines options for Zoom augment

    Attributes
    ----------
    range_low, range_high: float
        Range of random factor by which to zoom in/out, usually
        smaller than 1 and around 1
    after_int_aug: bool
        Perform Histogram augmentation after Intensity augment

    Notes
    -----
    Assumes data in range [0, 1]

    shift = (1.0 - data)
    data = data * factor + (data * shift) * (1.0 - factor)
    """
    range_low = attr.ib(type=float, default=0.1)
    range_high = attr.ib(type=float, default=1.0)
    after_int_aug = attr.ib(type=bool, default=True)


@attr.s(kw_only=True)
class NormalizeConfig:
    """Defines options for Data Normalization

    Attributes
    ----------
    type: str
        Which kind of data normalization/standardization to use:
        [None or default, minmax, percminmax, mean, median]
    norm_bounds: list of int
        Used for minmax norm,  expects [min, max] used to norm data
        E.g. if int16 is used but data is only in range [2000, 7500]
    perc_min, perc_max: str
        Which percentile to use for data normalization, have to be
        precomputed and stored in data_config.toml per sample
        E.g.
        [stats]
        perc0_01 = 2036
        perc3 = 2087
        perc99_8 = 4664
        perc99_99 = 7206
    """
    type = attr.ib(type=str, default=None)
    norm_bounds = attr.ib(type=List[int], default=None)
    perc_min = attr.ib(type=str, default=None)
    perc_max = attr.ib(type=str, default=None)


@attr.s(kw_only=True)
class _AugmentConfig:
    """Combines different augments into one section in configuration

    By default all augmentations are turned off if not set otherwise.
    Use one of the derived classes below depending on the use case
    (tracking vs cell state/cycle classifier)

    Notes
    -----
    If implementing new augments, add them to this list.
    """
    elastic = attr.ib(converter=ensure_cls(AugmentElasticConfig),
                      default=None)
    shift = attr.ib(converter=ensure_cls(AugmentShiftConfig),
                    default=None)
    shuffle_channels = attr.ib(type=bool, default=False)
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
class AugmentTrackingConfig(_AugmentConfig):
    """Specialized class for augmentation for tracking

    Attributes
    ----------
    reject_empty_prob: float
        Probability that completely empty patches are discarded
    divisions: float
        Choose (x*100)% of patches such that they include a division (e.g. 0.25)
    point_balance_radius: int
        Defines radius per point, the more other points within radius
        the lower the probability that point is picked, helps to avoid
        oversampling of dense regions
    """
    reject_empty_prob = attr.ib(type=float, default=1.0) # (default=1.0?)
    divisions = attr.ib(type=float, default=0.0)
    point_balance_radius = attr.ib(type=int, default=1)


@attr.s(kw_only=True)
class AugmentCellCycleConfig(_AugmentConfig):
    """Specialized class for augmentation for cell state/cycle classifier

    Attributes
    ----------
    min_key, max_key: str
        Which statistic to use for normalization, have to be
        precomputed and stored in data_config.toml per sample
        E.g.
        [stats]
        min = 1874
        max = 65535
        mean = 2260
        std = 282
        perc0_01 = 2036
        perc3 = 2087
        perc99_8 = 4664
        perc99_99 = 7206
    norm_min, norm_max: int
        Default values used it min/max_key do not exist
    jitter:
        See AugmentJitterConfig, shift selected point slightly

    Notes
    -----
    TODO: move data normalization info outside
    """
    min_key = attr.ib(type=str, default=None)
    max_key = attr.ib(type=str, default=None)
    norm_min = attr.ib(type=int, default=None)
    norm_max = attr.ib(type=int, default=None)
    jitter = attr.ib(converter=ensure_cls(AugmentJitterConfig),
                     default=None)
