import attr


@attr.s(kw_only=True)
class GeneralConfig:
    setup = attr.ib(type=str)
    # TODO: use post_init to set setup = basename(setup_dir)?
    setup_dir = attr.ib(type=str)
    db_host = attr.ib(type=str)
    sample = attr.ib(type=str)
    db_name = attr.ib(type=str, default=None)
    singularity_image = attr.ib(type=str, default=None)
    sparse = attr.ib(type=bool, default=True)
    seed = attr.ib(type=int)
    logging = attr.ib(type=int)
