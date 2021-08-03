import attr


@attr.s
class JobConfig:
    num_workers = attr.ib(type=int)
    queue = attr.ib(type=str)
    lab = attr.ib(type=str, default=None)
    singularity_image = attr.ib(type=str, default=None)
    local = attr.ib(type=bool, default=False)
