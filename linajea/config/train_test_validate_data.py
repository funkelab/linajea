"""Configuration used to define a dataset for training/test/validation

A dataset can consist of multiple samples (data sources). For Training
TrainData has to be defined. If the automated data selection functions
are used define ValData and TestData (support multiple samples),
otherwise define InferenceData (only a single sample, data source)
"""
from copy import deepcopy
from typing import List

import attr
import daisy

from .data import (DataSourceConfig,
                   DataROIConfig)
from .utils import (ensure_cls,
                    ensure_cls_list)


@attr.s(kw_only=True)
class _DataConfig():
    """Defines a base class for the definition of a data set

    Attributes
    ----------
    data_sources: list of DataSourceConfig
        List of data sources, can also only have a single element
    voxel_size: list of int, optional
    roi: DataROIConfig, optional
    group: str, optional
        voxel_size, roi and group can be set on the data source level
        and on the data set level. If set on the data set level, they
        the same values are used for all data sources
    """
    data_sources = attr.ib(converter=ensure_cls_list(DataSourceConfig))
    voxel_size = attr.ib(type=List[int], default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
    group = attr.ib(type=str, default=None)

    def __attrs_post_init__(self):
        """Validate the supplied parameters and try to fix missing ones

        For every data source:
        The ROI has to be set.
        If it is not set, check if it has been set on the data set level
            If yes, use this value.
            If no, use the ROI of the data file
        The ROI cannot be larger than the ROI of the data file
        The voxel size has to be set.
        If it is not set, do the same as for the ROI
        The group/array to be used has to be set.
        If it is not set, do the same as for the ROI

        The voxel size has to be identical for all data sources
        """
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
            d.roi.offset = roi.get_offset()
            d.roi.shape = roi.get_shape()
            if d.voxel_size is None:
                if self.voxel_size is None:
                    d.voxel_size = d.datafile.file_voxel_size
                    self.voxel_size = d.voxel_size
                else:
                    d.voxel_size = self.voxel_size
            if d.datafile.group is None:
                if self.group is None:
                    raise ValueError("no {group} supplied for data source")
                d.datafile.group = self.group

        assert all(ds.voxel_size == self.data_sources[0].voxel_size
                   for ds in self.data_sources), \
            "data sources with varying voxel_size not supported"


@attr.s(kw_only=True)
class TrainDataTrackingConfig(_DataConfig):
    """Defines a specialized class for the definition of a training data set
    """
    data_sources = attr.ib(converter=ensure_cls_list(DataSourceConfig))

    @data_sources.validator
    def _check_train_data_source(self, attribute, value):
        """train data source has to use datafiles and cannot have a database"""
        for ds in value:
            if ds.db_name is not None:
                raise ValueError("train data_sources must not have a db_name")


@attr.s(kw_only=True)
class TestDataTrackingConfig(_DataConfig):
    """Defines a specialized class for the definition of a test data set

    Attributes
    ----------
    checkpoint: int
        Which checkpoint of the trained model should be used?
    cell_score_threshold: float
        What is the minimum score of object/node candidates?
    """
    checkpoint = attr.ib(type=int, default=None)
    cell_score_threshold = attr.ib(type=float, default=None)


@attr.s(kw_only=True)
class InferenceDataTrackingConfig():
    """Defines a class for the definition of an inference data set

    An inference data set has only a single data source.
    If the getNextInferenceData facility is used for inference, it is
    set automatically based on the values for validate/test data and
    the current step in the pipeline.
    Otherwise it has to be set manually (and instead of validate/test)
    data.

    Attributes
    ----------
    data_source: DataSourceConfig
        Which data source should be used?
    checkpoint: int
        Which checkpoint of the trained model should be used?
    cell_score_threshold: float
        What is the minimum score of object/node candidates?
    """
    data_source = attr.ib(converter=ensure_cls(DataSourceConfig))
    checkpoint = attr.ib(type=int, default=None)
    cell_score_threshold = attr.ib(type=float, default=None)

    def __attrs_post_init__(self):
        """Try to fix ROI/voxel size if needed

        If a data file is set, and ROI or voxel size are not set, try
        to set it based on file info
        If data file is not set, and ROI or voxel size are not set and
        database does not contain the respective information an error
        will be thrown later in the pipeline.
        """
        d = self.data_source
        if d.datafile is not None:
            if d.voxel_size is None:
                d.voxel_size = d.datafile.file_voxel_size
            if d.roi is None:
                d.roi = d.datafile.file_roi


@attr.s(kw_only=True)
class ValidateDataTrackingConfig(_DataConfig):
    """Defines a specialized class for the definition of a validation data set

    Attributes
    ----------
    checkpoints: list of int
        Which checkpoints of the trained model should be used?
    cell_score_threshold: float
        What is the minimum score of object/node candidates?

    Notes
    -----
    Computes the results for every checkpoint
    """
    checkpoints = attr.ib(type=List[int], default=[None])
    cell_score_threshold = attr.ib(type=float, default=None)
