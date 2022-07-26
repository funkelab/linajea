"""Configuration used to define U-Net architecture
"""

from typing import List

import attr

from .utils import (_check_nd_shape,
                    _int_list_validator,
                    _list_int_list_validator,
                    _check_possible_nested_lists)


@attr.s(kw_only=True)
class UnetConfig:
    """Defines U-Net architecture

    Attributes
    ----------
    train_input_shape, predict_input_shape: list of int
        4d (t+3d) shape (in voxels) of input to network; should be as big as
        your GPU memory permits
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
        test/inference; should be a bit smaller than the minimal distance
        between two cell centers.
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
        Use very small weight for pixels lower than cutoff in gt map.
        Should only be enabled for dense GT and disabled for sparse GT, puts a
        loss on background pixels.
    cell_indicator_cutoff: float
        Cutoff values for weight, in gt_cell_indicator map
    train_only_cell_indicator: bool
        Only train cell indicator network, not movement vectors

    Notes
    -----
    In general, `shape` is in voxels, `size` is in world units, the input
    to the network has to be specified in voxels.
    (though kernel_size is still in voxels)
    """
    train_input_shape = attr.ib(type=List[int],
                                validator=[_int_list_validator,
                                           _check_nd_shape(4)])
    predict_input_shape = attr.ib(type=List[int],
                                  validator=[_int_list_validator,
                                             _check_nd_shape(4)])
    num_fmaps = attr.ib(type=int, default=12)
    fmap_inc_factors = attr.ib(type=int, default=4)
    downsample_factors = attr.ib(type=List[List[int]],
                                 validator=_list_int_list_validator,
                                 default=[[1, 2, 2],
                                          [1, 2, 2],
                                          [2, 2, 2]])
    kernel_size_down = attr.ib(type=List[List[int]],
                               validator=_check_possible_nested_lists,
                               default=[[[3, 3, 3, 3], [3, 3, 3, 3]],
                                        [[3, 3, 3, 3], [3, 3, 3]],
                                        [[3, 3, 3], [3, 3, 3]],
                                        [[3, 3, 3], [3, 3, 3]]])
    kernel_size_up = attr.ib(type=List[List[int]],
                             validator=_check_possible_nested_lists,
                             default=[[[3, 3, 3], [3, 3, 3]],
                                      [[3, 3, 3], [3, 3, 3]],
                                      [[3, 3, 3], [3, 3, 3]]])
    upsampling = attr.ib(type=str, default="trilinear",
                         validator=attr.validators.optional(
                             attr.validators.in_([
                                 "transposed_conv",
                                 "sep_transposed_conv",  # depthwise+pixelwise
                                 "resize_conv",
                                 "uniform_transposed_conv",
                                 "pixel_shuffle",
                                 "trilinear",  # aka 3d bilinear
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
    cell_indicator_weighted = attr.ib(type=float, default=None)
    cell_indicator_cutoff = attr.ib(type=float, default=0.01)
    train_only_cell_indicator = attr.ib(type=bool, default=False)
