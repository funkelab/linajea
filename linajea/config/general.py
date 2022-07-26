"""General configuration parameters

Parameters that are used in all steps, e.g., were is the experimental setup
stored, which database host to use, which logging level should be used.
"""
import os

import attr


@attr.s(kw_only=True)
class GeneralConfig:
    """Defines general configuration parameters

    Attributes
    ----------
    setup_dir: str
        Where the experiment data is/should be stored (e.g. model
        checkpoints, configuration files etc.), can be None if working
        with existing CandidateDatabase (no training/prediction)
    db_host: str
        Address of the mongodb server, by default a local server is assumed
    sparse: bool
        Is the ground truth sparse (not every instance is annotated)
    logging: int
        Which python logging level should be used:
        (10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR)
    seed: int
        Which random seed to use, for replication of experiments,
        experimental, not used everywhere yet
    tag: str, optional
        Tag for experiment, can be used for debugging purposes
    singularity_image: str, optional
        Which singularity image to use to run code, optional, not required
        if a conda/virtual environment is used.
    """
    # set via post_init hook
    setup_dir = attr.ib(type=str, default=None)
    db_host = attr.ib(type=str, default="mongodb://localhost:27017")
    sparse = attr.ib(type=bool, default=True)
    logging = attr.ib(type=int, default=20)
    seed = attr.ib(type=int, default=42)
    tag = attr.ib(type=str, default=None)
    singularity_image = attr.ib(type=str, default=None)
