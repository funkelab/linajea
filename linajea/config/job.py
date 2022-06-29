"""Configuration used to define where and how a job is to be executed
"""
import attr


@attr.s
class JobConfig:
    """Defines the configuration for where and how a job should be executed

    Attributes
    ----------
    num_workers: int
        How many processes/workers to use
    queue: str
        Which HPC job queue to use (for lsf)
    lab:
        Under which budget/account the job should run (for lsf)
    singularity_image: str
        Which singularity image should be used, deprecated
    run_on: str
        on which type of (hpc) system should the job be run
        one of: local, lsf, slurm, gridengine
        tested: local, lsf
        experimental: slurm, gridengine

    """
    num_workers = attr.ib(type=int, default=1)
    queue = attr.ib(type=str, default="local")
    lab = attr.ib(type=str, default=None)
    singularity_image = attr.ib(type=str, default=None)
    run_on = attr.ib(type=str, default="local",
                     validator=attr.validators.in_(["local",
                                                    "lsf",
                                                    "slurm",
                                                    "gridengine"]))
