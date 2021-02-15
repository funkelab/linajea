import attr


@attr.s
class GeneralConfig:
    setup = attr.ib(type=str)
    setups_dir = attr.ib(type=str)
    db_host = attr.ib(type=str)
    sample = attr.ib(type=str)
    db_name = attr.ib(type=str, default=None)
    singularity_image = attr.ib(type=str, default=None)
    sparse = attr.ib(type=bool, default=True)