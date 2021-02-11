import attr


@attr.s
class PredictConfig:
    cell_score_threshold = attr.ib(type=float)
    num_workers = attr.ib(type=int)
    processes_per_worker = attr.ib(type=int)
