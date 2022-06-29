"""Configuration used to define a Data Sample (DataSource)

One DataSource has to be defined for each sample. Typically one source
is defined by one file. If no training and prediction should be
performed it can alternatively be defined by a database (db_name).
Otherwise the database will be uniquely identified based on file and
prediction parameters.
"""
import os
from typing import List

import attr
import daisy
import zarr

from .utils import (ensure_cls,
                    load_config,
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
    group: str
        Which array/group in file to use (for n5/zarr/hdf)
    file_roi: DataROIConfig
        Size of data/ROI contained in file, determined automatically
    file_voxel_size: list of int
        Voxel size of data in file, determined automatically
    file_track_range: list of int
        deprecated
    """
    filename = attr.ib(type=str)
    group = attr.ib(type=str, default=None)
    file_roi = attr.ib(default=None)
    file_voxel_size = attr.ib(default=None)
    file_track_range = attr.ib(type=List[int], default=None)

    def __attrs_post_init__(self):
        """Read voxel size and ROI from file

        If n5/zarr, info should be contained in meta data.
        Otherwise location should contain a data_config.toml file with
        the respective information.

        Notes
        -----
        Example for data_config.toml file:
        [general]
        zarr_file = "emb.zarr"
        mask_file = "emb_mask.hdf"
        shape = [425, 41, 512, 512]
        resolution = [1, 5, 1, 1]
        offset = [0, 0, 0, 0]
        tracks_file = "mskcc_emb_tracks.txt"
        daughter_cells_file = "mskcc_emb_tracks_daughters.txt"

        [stats]
        dtype = "uint16"
        min = 1874
        max = 655535
        """
        if os.path.splitext(self.filename)[1] in (".n5", ".zarr"):
            dataset = daisy.open_ds(self.filename, self.group)

            self.file_voxel_size = dataset.voxel_size
            self.file_roi = DataROIConfig(offset=dataset.roi.get_offset(),
                                          shape=dataset.roi.get_shape())  # type: ignore
        else:
            filename = self.filename
            is_polar = "polar" in filename
            if is_polar:
                filename = filename.replace("_polar", "")
            if os.path.isdir(filename):
                data_config = load_config(os.path.join(filename,
                                                       "data_config.toml"))
            else:
                data_config = load_config(
                    os.path.join(os.path.dirname(filename),
                                 "data_config.toml"))
            self.file_voxel_size = data_config['general']['resolution']
            self.file_roi = DataROIConfig(
                offset=data_config['general']['offset'],
                shape=[s*v for s,v in zip(data_config['general']['shape'],
                                          self.file_voxel_size)])  # type: ignore
            self.file_track_range = data_config['general'].get('track_range')
            if self.group is None:
                self.group = data_config['general']['group']


@attr.s(kw_only=True)
class DataDBMetaConfig:
    """Defines a configuration uniquely identifying a database

    Attributes
    ----------
    setup_dir: str
        Name of the used setup
    checkpoint: int
        Which model checkpoint was used for prediction
    cell_score_threshold: float
        Which cell score threshold was used during prediction
    voxel_size: list of int
        What is the voxel size of the data?

    Notes
    -----
    TODO add roi? remove voxel_size?
    """
    setup_dir = attr.ib(type=str, default=None)
    checkpoint = attr.ib(type=int)
    cell_score_threshold = attr.ib(type=float)
    voxel_size = attr.ib(type=List[int], default=None)


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
    divisionsfile, daughtersfile: str, optional
        During training, divisions can optionally be sampled more often.
        If enabled in training configuration, tracks in `daughtersfile`
        are sampled for this purpose. If `divisionsfile` is set, it should
        contain the cells in the temporal context around each entry in
        `daughtersfile`, otherwise `tracksfile` is used.
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
    divisionsfile = attr.ib(type=str, default=None)
    daughtersfile = attr.ib(type=str, default=None)
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
                self.db_name is not None,
                "please specify either a file source (datafile) "
                "or a database source (db_name)")
