from copy import deepcopy
import os
from typing import List

import attr
import daisy

from .data import (DataSourceConfig,
                   DataDBMetaConfig,
                   DataROIConfig)
from .utils import (ensure_cls,
                    ensure_cls_list)


@attr.s(kw_only=True)
class DataConfig():
    data_sources = attr.ib(converter=ensure_cls_list(DataSourceConfig))
    def __attrs_post_init__(self):
        for d in self.data_sources:
            if d.roi is None:
                if self.roi is None:
                    # if roi/voxelsize not set, use info from file
                    d.roi = deepcopy(d.datafile.file_roi)
                else:
                    # if data sample specific roi/voxelsize not set,
                    # use general one
                    d.roi = deepcopy(self.roi)
            file_roi = daisy.Roi(offset=d.datafile.file_roi.offset,
                                 shape=d.datafile.file_roi.shape)
            roi = daisy.Roi(offset=d.roi.offset,
                            shape=d.roi.shape)
            roi = roi.intersect(file_roi)
            if d.datafile.file_track_range is not None:
                if isinstance(d.datafile.file_track_range, dict):
                    if os.path.isdir(d.datafile.filename):
                        # TODO check this for training
                        begin_frame, end_frame = list(d.datafile.file_track_range.values())[0]
                        for _, v in d.datafile.file_track_range.items():
                            if v[0] < begin_frame:
                                begin_frame = v[0]
                            if v[1] > end_frame:
                                end_frame = v[1]
                    else:
                        begin_frame, end_frame = d.datafile.file_track_range[
                            os.path.basename(d.datafile.filename)]
                else:
                    begin_frame, end_frame = d.datafile.file_track_range
                track_range_roi = daisy.Roi(
                    offset=[begin_frame, None, None, None],
                    shape=[end_frame-begin_frame+1, None, None, None])
                roi = roi.intersect(track_range_roi)
                # d.roi.offset[0] = max(begin_frame, d.roi.offset[0])
                # d.roi.shape[0] = min(end_frame - begin_frame + 1,
                #                      d.roi.shape[0])
            d.roi.offset = roi.get_offset()
            d.roi.shape = roi.get_shape()
            if d.voxel_size is None:
                if self.voxel_size is None:
                    d.voxel_size = d.datafile.file_voxel_size
                else:
                    d.voxel_size = self.voxel_size
            if d.datafile.group is None:
                if self.group is None:
                    raise ValueError("no {group} supplied for data source")
                d.datafile.group = self.group

        assert all(ds.voxel_size == self.data_sources[0].voxel_size
                   for ds in self.data_sources), \
                       "data sources with varying voxel_size not supported"

    voxel_size = attr.ib(type=List[int], default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
    group = attr.ib(type=str, default=None)


@attr.s(kw_only=True)
class TrainDataTrackingConfig(DataConfig):
    data_sources = attr.ib(converter=ensure_cls_list(DataSourceConfig))
    @data_sources.validator
    def _check_train_data_source(self, attribute, value):
        for ds in value:
            if ds.db_name is not None:
                raise ValueError("train data_sources must not have a db_name")


@attr.s(kw_only=True)
class TestDataTrackingConfig(DataConfig):
    checkpoint = attr.ib(type=int, default=None)
    cell_score_threshold = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class InferenceDataTrackingConfig():
    data_source = attr.ib(converter=ensure_cls(DataSourceConfig))
    checkpoint = attr.ib(type=int, default=None)
    cell_score_threshold = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class ValidateDataTrackingConfig(DataConfig):
    checkpoints = attr.ib(type=List[int])
    cell_score_threshold = attr.ib(type=float, default=None)



@attr.s(kw_only=True)
class DataCellCycleConfig(DataConfig):
    use_database = attr.ib(type=bool)
    db_meta_info = attr.ib(converter=ensure_cls(DataDBMetaConfig), default=None)
    skip_predict = attr.ib(type=bool, default=False)
    force_predict = attr.ib(type=bool, default=False)


@attr.s(kw_only=True)
class TrainDataCellCycleConfig(DataCellCycleConfig):
    pass


@attr.s(kw_only=True)
class TestDataCellCycleConfig(DataCellCycleConfig):
    checkpoint = attr.ib(type=int)
    prob_threshold = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class ValidateDataCellCycleConfig(DataCellCycleConfig):
    checkpoints = attr.ib(type=List[int])
    prob_thresholds = attr.ib(type=List[float], default=None)


@attr.s(kw_only=True)
class InferenceDataCellCycleConfig():
    data_source = attr.ib(converter=ensure_cls(DataSourceConfig))
    checkpoint = attr.ib(type=int, default=None)
    prob_threshold = attr.ib(type=float)
    use_database = attr.ib(type=bool)
    db_meta_info = attr.ib(converter=ensure_cls(DataDBMetaConfig), default=None)
    force_predict = attr.ib(type=bool, default=False)
