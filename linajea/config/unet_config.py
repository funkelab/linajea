import attr
from typing import List

from .job import JobConfig
from .utils import (ensure_cls,
                    _check_nd_shape,
                    _int_list_validator,
                    _list_int_list_validator)


@attr.s(kw_only=True)
class UnetConfig:
    path_to_script = attr.ib(type=str)
    # shape -> voxels, size -> world units
    train_input_shape = attr.ib(type=List[int],
                                validator=[_int_list_validator,
                                           _check_nd_shape(4)])
    predict_input_shape = attr.ib(type=List[int],
                                  validator=[_int_list_validator,
                                             _check_nd_shape(4)])
    fmap_inc_factors = attr.ib(type=int)
    downsample_factors = attr.ib(type=List[List[int]],
                                 validator=_list_int_list_validator)
    kernel_size_down = attr.ib(type=List[List[int]],
                               validator=_list_int_list_validator)
    kernel_size_up = attr.ib(type=List[List[int]],
                             validator=_list_int_list_validator)
    upsampling = attr.ib(type=str, default="uniform_transposed_conv",
                         validator=attr.validators.in_([
                             "transposed_conv", "resize_conv",
                             "uniform_transposed_conv"]))
    constant_upsample = attr.ib(type=bool)
    nms_window_shape = attr.ib(type=List[int],
                               validator=[_int_list_validator,
                                          _check_nd_shape(3)])
    nms_window_shape_test = attr.ib(
        type=List[int], default=None,
        validator=attr.validators.optional([_int_list_validator,
                                            _check_nd_shape(3)]))
    average_vectors = attr.ib(type=bool, default=False)
    unet_style = attr.ib(type=str, default='split')
    num_fmaps = attr.ib(type=int)
    cell_indicator_weighted = attr.ib(type=bool, default=True)
