"""Configuration used for evaluation
"""
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
    ignore_one_off_div_errors = attr.ib(type=bool, default=False)
    fn_div_count_unconnected_parent = attr.ib(type=bool, default=True)

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
