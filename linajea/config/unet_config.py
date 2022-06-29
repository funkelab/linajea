"""Configuration used to define U-Net architecture
"""

from typing import List

import attr

from .job import JobConfig
from .utils import (ensure_cls,
                    _check_nd_shape,
                    _int_list_validator,
                    _list_int_list_validator,
                    _check_possible_nested_lists)


@attr.s(kw_only=True)
class UnetConfig:
    """Defines U-Net architecture

    Attributes
    ----------
    path_to_script: str
        Location of training script
    train_input_shape, predict_input_shape: list of int
        4d (t+3d) shape (in voxels) of input to network
    fmap_inc_factors: list of int
        By which factor to increase number of channels during each
        pooling step, number of values depends on number of pooling
        steps
    downsample_factors: list of list of int
        By which factor to downsample during pooling, one value per
        dimension per pooling step
    kernel_size_down, kernel_size_up: list of list of int
        Size of convolutional kernels, length of outer list depends
        on number of pooling steps, length of inner list depends on
        number of convolutions per step, both for encoder (down) and
        decoder (up) path
    upsampling: str
        What kind of upsampling function should be used, one of
        ["transposed_conv",
         "sep_transposed_conv" (=depthwise + pixelwise),
         "resize_conv",
         "uniform_transposed_conv",
         "pixel_shuffle",
         "trilinear" (= 3d bilinear),
         "nearest"]
    constant_upsample: bool
        Use nearest neighbor upsampling, deprecated,
        overwrites upsampling
    nms_window_shape, nms_window_shape_test: list of int
        Size of non-max suppression window used to extract maxima of
        cell indicator map, can optionally be different for
        test/inference
    average_vectors: bool
        Compute average movement vector in 3x3x3 window around maximum
    unet_style: str
        Style of network used for cell indicator and vectors, one of
        ["single": a single network for both, only last layer different,
         "split": two completely separate networks,
         "multihead": one encoder, separate decoders]
    num_fmaps: int
        Number of channels to create in first convolution
    cell_indicator_weighted: bool
        Use very small weight for pixels lower than cutoff in gt map
    cell_indicator_cutoff: float
        Cutoff values for weight, in gt_cell_indicator map
    chkpt_parents: str
        Not used, path to model checkpoint just for movement vectors
    chkpt_cell_indicator
        Not used, path to model checkpoint just for cell indicator
    latent_temp_conv: bool
        Apply temporal/4d conv not in the beginning but at bottleneck
    train_only_cell_indicator: bool
        Only train cell indicator network, not movement vectors
    """
    path_to_script = attr.ib(type=str, default=None)
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
                               default=None,
                               validator=_check_possible_nested_lists)
    kernel_size_up = attr.ib(type=List[List[int]],
                             default=None,
                             validator=_check_possible_nested_lists)
    upsampling = attr.ib(type=str, default=None,
                         validator=attr.validators.optional(attr.validators.in_([
                             "transposed_conv",
                             "sep_transposed_conv", # depthwise + pixelwise
                             "resize_conv",
                             "uniform_transposed_conv",
                             "pixel_shuffle",
                             "trilinear", # aka 3d bilinear
                             "nearest"
                         ])))
    constant_upsample = attr.ib(type=bool, default=None)
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
    cell_indicator_cutoff = attr.ib(type=float, default=0.01)
    chkpt_parents = attr.ib(type=str, default=None)
    chkpt_cell_indicator = attr.ib(type=str, default=None)
    latent_temp_conv = attr.ib(type=bool, default=False)
    train_only_cell_indicator = attr.ib(type=bool, default=False)
