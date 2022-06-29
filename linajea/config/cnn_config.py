"""Configuration used to define CNN architecture

Currently supports VGG/ResNet and EfficientNet style networks.
"""
from typing import List

import attr

from .job import JobConfig
from .utils import (ensure_cls,
                    _check_nd_shape,
                    _int_list_validator,
                    _list_int_list_validator)


@attr.s(kw_only=True)
class _CNNConfig:
    """Defines base network class with common parameters

    Attributes
    ----------
    path_to_script: str
        Location of training script
    input_shape: list of int
        4d (t+3d) shape (in voxels) of input to network
    pad_raw: list of int
        Padding added to sample to allow for sampling of cells at the
        boundary, 4d (t+3d)
    activation: str
        Activation function used by the network, given string has to be
        supported by the used framework
    padding: str
        SAME or VALID, type of padding used in convolutions
    merge_time_voxel_size: int
        When to use 4d convolutions to merge time dimension into
        spatial dimensions, based on voxel_size
        Voxel_size doubles after each pooling layer,
        if voxel_size >= merge_time_voxel_size, insert 4d conv
    use_dropout: bool
        If dropout (with rate 0.1) is used in conv layers
    use_bias: bool
        Whether a bias variable is used in conv layer
    use_global_pool: bool
        Whether global pooling instead of fully-connected layers is
        used at the end of the network
    use_conv4d: bool
        Whether 4d convolutions are used for 4d data or
        fourth dimension is interpreted as channels
        (similar to RGB data)
    num_classes: int
        Number of output classes
    classes: list of str
        Name of each class
    class_ids: list of int
        ID for each class
    class_sampling_weights: list of int
        Ratio with which each class is sampled per batch,
        does not have to be normalized to sum to 1
    network_type: str
        Type of netowrk used, one of ["vgg", "resnet", "efficientnet"]
    make_isotropic: bool
        Do not pool anisotropic dimension until data is approx isotropic
    regularizer_weight: float
        Weight for l2 weight regularization
    with_polar: bool
        For C. elegans, should polar body class be included?
    focal_loss: bool
        Whether to use focal loss instead of plain cross entropy
    classify_dataset: bool
        For testing purposes, ignore label and try to decide which
        data sample a cell is orginated from
    """
    path_to_script = attr.ib(type=str)
    # shape -> voxels, size -> world units
    input_shape = attr.ib(type=List[int],
                          validator=[_int_list_validator,
                                     _check_nd_shape(4)],
                          default=[5, 32, 32, 32])
    pad_raw = attr.ib(type=List[int],
                      validator=[_int_list_validator,
                                 _check_nd_shape(4)],
                      default=[3, 30, 30, 30])
    activation = attr.ib(type=str, default='relu')
    padding = attr.ib(type=str, default='SAME')
    merge_time_voxel_size = attr.ib(type=int, default=1)
    use_dropout = attr.ib(type=bool, default=True)
    use_batchnorm = attr.ib(type=bool, default=True)
    use_bias = attr.ib(type=bool, default=True)
    use_global_pool = attr.ib(type=bool, default=True)
    use_conv4d = attr.ib(type=bool, default=True)
    num_classes = attr.ib(type=int, default=3)
    classes = attr.ib(type=List[str], default=["daughter", "mother", "normal"])
    class_ids = attr.ib(type=List[int], default=[2, 1, 0])
    class_sampling_weights = attr.ib(type=List[int], default=[6, 2, 2, 2])
    network_type = attr.ib(
        type=str,
        validator=attr.validators.in_(["vgg", "resnet", "efficientnet"]))
    make_isotropic = attr.ib(type=int, default=False)
    regularizer_weight = attr.ib(type=float, default=None)
    with_polar = attr.ib(type=int, default=False)
    focal_loss = attr.ib(type=bool, default=False)
    classify_dataset = attr.ib(type=bool, default=False)

@attr.s(kw_only=True)
class VGGConfig(_CNNConfig):
    """Specialized class for VGG style networks

    Attributes
    ----------
    num_fmaps: int
        Number of channels to create in first convolution
    fmap_inc_factors: list of int
        By which factor to increase number of channels during each
        pooling step, number of values depends on number of pooling
        steps
    downsample_factors: list of list of int
        By which factor to downsample during pooling, one value per
        dimension per pooling step
    kernel_sizes: list of list of int
        Size of convolutional kernels, length of outer list depends
        on number of pooling steps, length of inner list depends on
        number of convolutions per step
    net_name: str
        Name of network
    """
    num_fmaps = attr.ib(type=int, default=12)
    fmap_inc_factors = attr.ib(type=List[int],
                               validator=_int_list_validator,
                               default=[2, 2, 2, 1])
    downsample_factors = attr.ib(type=List[List[int]],
                                 validator=_list_int_list_validator,
                                 default=[[2, 2, 2],
                                          [2, 2, 2],
                                          [2, 2, 2],
                                          [1, 1, 1]])
    kernel_sizes = attr.ib(type=List[List[int]],
                           validator=_list_int_list_validator,
                           default=[[3, 3],
                                    [3, 3],
                                    [3, 3, 3, 3],
                                    [3, 3, 3, 3]])
    fc_size = attr.ib(type=int, default=512)
    net_name = attr.ib(type=str, default="vgg")


@attr.s(kw_only=True)
class ResNetConfig(_CNNConfig):
    """Specialized class for ResNet style networks

    Attributes
    ----------
    net_name:
        Name of network
    resnet_size: str
        Size of network, one of ["18", "34", "50", "101", None]
    num_blocks: list of int
        If resnet_size is None, use this many residual blocks per step
    use_bottleneck:
        If resnet_size is None, if bottleneck style blocks are used
    num_fmaps: str
        Number of feature maps used per step (typically 4 steps)
    """
    net_name = attr.ib(type=str, default="resnet")
    resnet_size = attr.ib(type=str, default=None)
    num_blocks = attr.ib(type=List[int],
                         validator=_int_list_validator,
                         default=[2, 2, 2, 2])
    use_bottleneck = attr.ib(type=bool, default=False)
    num_fmaps = attr.ib(type=List[int],
                        validator=_int_list_validator,
                        default=[16, 32, 64, 96])


@attr.s(kw_only=True)
class EfficientNetConfig(_CNNConfig):
    """Specialized class for EfficientNet style networks

    Attributes
    ----------
    net_name:
        Name of network
    efficientnet_size:
        Which efficient net size to use,
        one of "B0" to "B10"
    """
    net_name = attr.ib(type=str, default="efficientnet")
    efficientnet_size = attr.ib(type=str, default="B0")
