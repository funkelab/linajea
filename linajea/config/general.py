import os

import attr


@attr.s(kw_only=True)
class GeneralConfig:
    # set via post_init hook
    # setup = attr.ib(type=str)
    setup_dir = attr.ib(type=str)
    db_host = attr.ib(type=str)
    # sample = attr.ib(type=str)
    db_name = attr.ib(type=str, default=None)
    singularity_image = attr.ib(type=str, default=None)
    sparse = attr.ib(type=bool, default=True)
    seed = attr.ib(type=int)
    logging = attr.ib(type=int)

    def __attrs_post_init__(self):
        self.setup = os.path.basename(self.setup_dir)
