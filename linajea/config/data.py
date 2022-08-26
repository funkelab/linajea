"""Configuration used to define a Data Sample (DataSource)

One DataSource has to be defined for each sample. Typically one source
is defined by one file. If no training and prediction should be
performed it can alternatively be defined by a database (db_name).
Otherwise the database will be uniquely identified based on file and
prediction parameters.
"""
from typing import List

import attr
import daisy

from .utils import (ensure_cls,
                    _check_nested_nd_shape,
                    _list_int_list_validator)


@attr.s(kw_only=True)
class DataROIConfig:
    """Defines a ROI (region of interest)

    Attributes
    ----------
    offset: list of int
        Offset relative to origin
    shape: list of int
        Shape (not end!) of region, in world coordinates
    """
    offset = attr.ib(type=List[int], default=None)
    shape = attr.ib(type=List[int], default=None)


@attr.s(kw_only=True)
class DataFileConfig:
    """Defines a data file

    Attributes
    ----------
    filename: str
        Path to data file/directory
    array: str
        Which array in file to use (for n5/zarr/hdf)
    file_roi: DataROIConfig
        Size of data/ROI contained in file, determined automatically
    file_voxel_size: list of int
        Voxel size of data in file, determined automatically
    """
    filename = attr.ib(type=str)
    array = attr.ib(type=str, default=None)
    file_roi = attr.ib(default=None)
    file_voxel_size = attr.ib(default=None)

    def __attrs_post_init__(self):
        """Read voxel size, ROI and attributes from file
        """
        dataset = daisy.open_ds(self.filename, self.array)

        self.file_voxel_size = dataset.voxel_size
        self.file_roi = DataROIConfig(
            offset=dataset.roi.get_offset(),
            shape=dataset.roi.get_shape())  # type: ignore


@attr.s(kw_only=True)
class DataSourceConfig:
    """Defines a complete data source

    Notes
    ----_
    - either datafile or db_name have to be specified
    - tracksfile etc have to be specified for training
    - gt_db_name has to be specified for evaluation
    - voxel_size/roi have to be specified on some level, code tries to
      figure it out based on multiple sources (specified in config, file, etc)
    Attributes
    ----------
    datafile: DataFileConfig
        Describes file data source
    tracksfile: str
        File containing the object tracks used during training
    db_name: str
        Database to be used as a data source for tracking or as a
        destination for prediction. If not set during prediction, name
        will be set automatically.
    gt_db_name: str
        Database containing the ground truth annotations for evaluation.
    gt_db_name_polar: str
        Database containing the polar body ground truth annotations for
        evaluation.
    voxel_size: list of int
        Voxel size of this data source. If multiple samples are used,
        they must have the same voxel size. If not set, tries to determine
        information automatically from file or database.
    roi: DataROIConfig
        ROI of this data source that should be used during processing.
    exclude_times: list of list of int
        Which time frame intervals within the given ROI should be excluded.
        Expects a list of two-element lists, each defining a range of frames.
    """
    datafile = attr.ib(converter=ensure_cls(DataFileConfig), default=None)
    tracksfile = attr.ib(type=str, default=None)
    db_name = attr.ib(type=str, default=None)
    gt_db_name = attr.ib(type=str, default=None)
    gt_db_name_polar = attr.ib(type=str, default=None)
    voxel_size = attr.ib(type=List[int], default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
    exclude_times = attr.ib(type=List[List[int]],
                            validator=attr.validators.optional(
                                [_list_int_list_validator,
                                 _check_nested_nd_shape(2)]),
                            default=None)

    def __attrs_post_init__(self):
        assert (self.datafile is not None or
                self.db_name is not None), \
                ("please specify either a file source (datafile) "
                 "or a database source (db_name)")
