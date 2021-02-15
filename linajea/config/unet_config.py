import attr
from typing import List

from .data import DataConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class UnetConfig:
    # shape -> voxels, size -> world units
    train_input_shape = attr.ib(type=List[int])
    predict_input_shape = attr.ib(type=List[int])
    fmap_inc_factors = attr.ib(type=int)
    downsample_factors = attr.ib(type=List[List[int]])
    kernel_size_down = attr.ib(type=List[List[int]])
    kernel_size_up = attr.ib(type=List[List[int]])
    upsampling = attr.ib(type=str, default="uniform_transposed_conv")
    constant_upsample = attr.ib(type=bool)
    nms_window_shape = attr.ib(type=List[int])
    average_vectors = attr.ib(type=bool, default=False)
    unet_style = attr.ib(type=str, default='split')
    num_fmaps = attr.ib(type=int)
    cell_indicator_weighted = attr.ib(type=bool, default=True)
