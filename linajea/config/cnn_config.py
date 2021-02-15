import attr
from typing import List

from .data import DataConfig
from .job import JobConfig
from .utils import ensure_cls


@attr.s(kw_only=True)
class CNNConfig:
    # shape -> voxels, size -> world units
    input_shape = attr.ib(type=List[int])
    pad_raw = attr.ib(type=List[int])
    activation = attr.ib(type=str, default='relu')
    padding = attr.ib(type=str, default='SAME')
    merge_time_voxel_size = attr.ib(type=int, default=1)
    use_dropout = attr.ib(type=bool, default=True)
    use_batchnorm = attr.ib(type=bool, default=True)
    use_bias = attr.ib(type=bool, default=True)
    use_global_pool = attr.ib(type=bool, default=True)
    use_conv4d = attr.ib(type=bool, default=True)
    num_classes = attr.ib(type=int, default=3)
    network_type = attr.ib(type=str, default="vgg")
    make_isotropic = attr.ib(type=int, default=False)


@attr.s(kw_only=True)
class VGGConfig(CNNConfig):
    num_fmaps = attr.ib(type=List[int], default=32)
    fmap_inc_factors = attr.ib(type=List[int])
    downsample_factors = attr.ib(type=List[List[int]])
    kernel_sizes = attr.ib(type=List[List[int]])
    fc_size = attr.ib(type=int)
    net_name = attr.ib(type=str, default="vgg")


@attr.s(kw_only=True)
class ResNetConfig(CNNConfig):
    net_name = attr.ib(type=str, default="resnet")
    resnet_size = attr.ib(type=str, default="18")
    num_blocks = attr.ib(default=None, type=List[int])
    use_bottleneck = attr.ib(default=None, type=bool)


@attr.s(kw_only=True)
class EfficientNetConfig(CNNConfig):
    net_name = attr.ib(type=str, default="efficientnet")
    efficientnet_size = attr.ib(type=str, default="B01")
