import attr
from attr import validators
from typing import List


@attr.s
class GeneralConfig:
    setup = attr.ib(type=str)
    iteration = attr.ib(type=int)
    sample = attr.ib(type=str)
    db_host = attr.ib(type=str)
    db_name = attr.ib(type=str)
    singularity_image = attr.ib(type=str)
    data_dir = attr.ib(type=str)
    data_file = attr.ib(type=str)
    setups_dir = attr.ib(type=str)
    frames = attr.ib(
            type=List[int],
            validator=validators.deep_iterable(
                member_validator=validators.instance_of(int),
                iterable_validator=validators.instance_of(list)))
    queue = attr.ib(type=str, default='gpu_rtx')
    lab = attr.ib(type=str, default='funke')
