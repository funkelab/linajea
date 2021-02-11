import attr
from typing import List


@attr.s
class DataConfig:
    filename = attr.ib(type=str)
    group = attr.ib(type=str, default=None)
    voxel_size = attr.ib(type=List[int], default=None)
    roi_offset = attr.ib(type=List[int], default=None)
    roi_shape = attr.ib(type=List[int], default=None)
