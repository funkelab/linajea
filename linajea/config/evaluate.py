"""Configuration used for evaluation
"""
from typing import List

import attr

from .data import DataROIConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class EvaluateParametersTrackingConfig:
    """Defines a set of evaluation parameters for Tracking

    Notes
    -----
    This set, in combination with the solving step parameters,
    identifies a solution uniquely. Only solutions with the same
    evaluation parameters (but varying solving parameters) can be
    compared directly.
    Allows, for instance, to evaluate the model in different ROIs.

    Attributes
    ----------
    matching_threshold: int
        How far can a GT annotation and a predicted object be apart but
        still be matched to each other.
    roi: DataROIConfig
        Which ROI should be evaluated?
    validation_score: bool
        Should the validation score be computed (additional metric)
    window_size: int
        What is the maximum window size for which the fraction of
        error-free tracklets should be computed?
    filter_polar_bodies: bool
        Should polar bodies be removed from the computed tracks?
        Requires cell state classifier predictions, removes objects with
        a high polar body score from tracks, does not load GT polar
        bodies.
    filter_polar_bodies_key: str
        If polar bodies should be filtered, which attribute in database
        node collection should be used
    filter_short_tracklets_len: int
        If positive, remove all tracks shorter than this many objects
    ignore_one_off_div_errors: bool
        Division annotations are often slightly imprecise. Due to the
        limited temporal resolution the exact moment a division happens
        cannnot always be determined accuratly. If the predicted division
        happens 1 frame before or after an annotated one, does not count
        it as an error.
    fn_div_count_unconnected_parent: bool
        If the parent of the mother cell of a division is missing, should
        this count as a division error (aside from the already counted FN
        edge error)
    """
    matching_threshold = attr.ib(type=int)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
    validation_score = attr.ib(type=bool, default=False)
    window_size = attr.ib(type=int, default=50)
    filter_polar_bodies = attr.ib(type=bool, default=None)
    filter_polar_bodies_key = attr.ib(type=str, default=None)
    filter_short_tracklets_len = attr.ib(type=int, default=-1)
    ignore_one_off_div_errors = attr.ib(type=bool, default=False)
    fn_div_count_unconnected_parent = attr.ib(type=bool, default=True)
    # deprecated
    frames = attr.ib(type=List[int], default=None)
    # deprecated
    frame_start = attr.ib(type=int, default=None)
    # deprecated
    frame_end = attr.ib(type=int, default=None)
    # deprecated
    limit_to_roi_offset = attr.ib(type=List[int], default=None)
    # deprecated
    limit_to_roi_shape = attr.ib(type=List[int], default=None)
    # deprecated
    sparse = attr.ib(type=bool)

    def __attrs_post_init__(self):
        if self.frames is not None and \
           self.frame_start is None and self.frame_end is None:
            self.frame_start = self.frames[0]
            self.frame_end = self.frames[1]
            self.frames = None

    def valid(self):
        return {key: val
                for key, val in attr.asdict(self).items()
                if val is not None}

    def query(self):
        params_dict_valid = self.valid()
        params_dict_none = {key: {"$exists": False}
                            for key, val in attr.asdict(self).items()
                            if val is None}
        query = {**params_dict_valid, **params_dict_none}
        return query


@attr.s(kw_only=True)
class _EvaluateConfig:
    """Base class for evaluation configuration

    Attributes
    ----------
    job: JobConfig
        HPC cluster parameters, default constructed (executed locally)
        if not supplied
    """
    job = attr.ib(converter=ensure_cls(JobConfig),
                  default=attr.Factory(JobConfig))


@attr.s(kw_only=True)
class EvaluateTrackingConfig(_EvaluateConfig):
    """Defines specialized class for configuration of tracking evaluation

    Attributes
    ----------
    from_scratch: bool
        Recompute solution even if it already exists
    parameters: EvaluateParametersTrackingConfig
        Which evaluation parameters to use
    """
    from_scratch = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=ensure_cls(EvaluateParametersTrackingConfig))


@attr.s(kw_only=True)
class EvaluateParametersCellCycleConfig:
    """Defines a set of evaluation parameters for the cell state classifier

    Attributes
    ----------
    matching_threshold: int
        How far can a GT annotation and a predicted object be apart but
        still be matched to each other.
    roi: DataROIConfig
        Which ROI should be evaluated?
    """
    matching_threshold = attr.ib()
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)


@attr.s(kw_only=True)
class EvaluateCellCycleConfig(_EvaluateConfig):
    """Defines specialized class for configuration of cell state evaluation

    Attributes
    ----------
    max_samples: int
        maximum number of samples to evaluate, deprecated
    metric: str
        which metric to use, deprecated
    one_off: bool
        Check for one-frame-off divisions (and do not count them as errors)
    prob_threshold: float
        Ignore predicted objects with a lower score
    dry_run: bool
        Do not write results to database
    find_fn: bool
        Do not run normal evaluation but locate missing objects/FN
    force_eval: bool
        Run evaluation even if results exist already, deprecated
    parameters: EvaluateParametersCellCycleConfig
        Which evaluation parameters to use
    """
    max_samples = attr.ib(type=int)
    metric = attr.ib(type=str)
    one_off = attr.ib(type=bool)
    prob_threshold = attr.ib(type=float)
    dry_run = attr.ib(type=bool)
    find_fn = attr.ib(type=bool)
    force_eval = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=ensure_cls(EvaluateParametersCellCycleConfig))
