import attr
from typing import List

from .job import JobConfig
from .utils import (ensure_cls,
                    _check_nd_shape,
                    _int_list_validator,
                    _list_int_list_validator)


@attr.s(kw_only=True)
class CNNConfig:
    path_to_script = attr.ib(type=str)
    # shape -> voxels, size -> world units
    input_shape = attr.ib(type=List[int],
                          validator=[_int_list_validator,
                                     _check_nd_shape(4)])
    pad_raw = attr.ib(type=List[int],
                      validator=[_int_list_validator,
                                 _check_nd_shape(4)])
    activation = attr.ib(type=str, default='relu')
    padding = attr.ib(type=str, default='SAME')
    merge_time_voxel_size = attr.ib(type=int, default=1)
    use_dropout = attr.ib(type=bool, default=True)
    use_batchnorm = attr.ib(type=bool, default=True)
    use_bias = attr.ib(type=bool, default=True)
    use_global_pool = attr.ib(type=bool, default=True)
    use_conv4d = attr.ib(type=bool, default=True)
    num_classes = attr.ib(type=int, default=3)
    classes = attr.ib(type=List[str])
    class_ids = attr.ib(type=List[int])
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
class VGGConfig(CNNConfig):
    num_fmaps = attr.ib(type=int)
    fmap_inc_factors = attr.ib(type=List[int],
                               validator=_int_list_validator)
    downsample_factors = attr.ib(type=List[List[int]],
                                 validator=_list_int_list_validator)
    kernel_sizes = attr.ib(type=List[List[int]],
                           validator=_list_int_list_validator)
    fc_size = attr.ib(type=int)
    net_name = attr.ib(type=str, default="vgg")


@attr.s(kw_only=True)
class ResNetConfig(CNNConfig):
    net_name = attr.ib(type=str, default="resnet")
    resnet_size = attr.ib(type=str, default="18")
    num_blocks = attr.ib(default=None, type=List[int],
                         validator=_int_list_validator)
    use_bottleneck = attr.ib(default=None, type=bool)
    num_fmaps = attr.ib(type=List[int],
                        validator=_int_list_validator)


@attr.s(kw_only=True)
class EfficientNetConfig(CNNConfig):
    net_name = attr.ib(type=str, default="efficientnet")
    efficientnet_size = attr.ib(type=str, default="B01")
