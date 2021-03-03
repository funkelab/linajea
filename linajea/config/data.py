import os
from typing import List

import attr

from linajea import load_config
from .utils import ensure_cls


@attr.s(kw_only=True)
class DataROIConfig:
    offset = attr.ib(type=List[int], default=None)
    shape = attr.ib(type=List[int], default=None)


@attr.s(kw_only=True)
class DataFileConfig:
    filename = attr.ib(type=str)
    group = attr.ib(type=str, default=None)
    file_roi = attr.ib(default=None)
    file_voxel_size = attr.ib(default=None)

    def __attrs_post_init__(self):
        if os.path.splitext(self.filename)[1] in (".n5", ".zarr"):
            if "nested" in self.group:
                store = zarr.NestedDirectoryStore(self.filename)
            else:
                store = self.filename
            container = zarr.open(store)
            attributes = zarr_container[self.group].attrs

            self.file_voxel_size = attributes.voxel_size
            self.file_roi = DataROIConfig(offset=attributes.offset,
                                          shape=attributes.shape)  # type: ignore
        else:
            data_config = load_config(os.path.join(self.filename,
                                                   "data_config.toml"))
            self.file_voxel_size = data_config['general']['resolution']
            self.file_roi = DataROIConfig(
                offset=data_config['general']['offset'],
                shape=[s*v for s,v in zip(data_config['general']['shape'],
                                          self.file_voxel_size)])  # type: ignore
            if self.group is None:
                self.group = data_config['general']['group']


@attr.s(kw_only=True)
class DataDBMetaConfig:
    setup_dir = attr.ib(type=str, default=None)
    checkpoint = attr.ib(type=int)
    cell_score_threshold = attr.ib(type=float)


@attr.s(kw_only=True)
class DataSourceConfig:
    datafile = attr.ib(converter=ensure_cls(DataFileConfig))
    db_name = attr.ib(type=str, default=None)
    gt_db_name = attr.ib(type=str, default=None)
    voxel_size = attr.ib(type=List[int], default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
