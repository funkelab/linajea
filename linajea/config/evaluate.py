from typing import List

import attr

from .data import DataROIConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class EvaluateParametersConfig:
    matching_threshold = attr.ib(type=int)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
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
class EvaluateConfig:
    job = attr.ib(converter=ensure_cls(JobConfig), default=None)


@attr.s(kw_only=True)
class EvaluateTrackingConfig(EvaluateConfig):
    from_scratch = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=ensure_cls(EvaluateParametersConfig))


@attr.s(kw_only=True)
class EvaluateParametersCellCycleConfig:
    matching_threshold = attr.ib()
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)


@attr.s(kw_only=True)
class EvaluateCellCycleConfig(EvaluateConfig):
    max_samples = attr.ib(type=int)
    metric = attr.ib(type=str)
    one_off = attr.ib(type=bool)
    prob_threshold = attr.ib(type=float)
    dry_run = attr.ib(type=bool)
    find_fn = attr.ib(type=bool)
    parameters = attr.ib(converter=ensure_cls(EvaluateParametersCellCycleConfig))
