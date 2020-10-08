from dataclasses import dataclass
from typing import List
from linajea import load_config


class LinajeaConfig:

    def __init__(self, config_file):
        config_dict = load_config(config_file)
        self.general = GeneralConfig(**config_dict['general'])
        self.predict = PredictConfig(**config_dict['predict'])
        self.extract = ExtractConfig(**config_dict['extract_edges'])
        self.evaluate = EvaluateConfig(**config_dict['evaluate'])


@dataclass
class GeneralConfig:
    setup: str
    iteration: int
    sample: str
    db_host: str
    db_name: str
    prediction_type: str
    singularity_image: str
    queue: str
    data_dir: str
    setups_dir: str
    frames: List[int]


@dataclass
class PredictConfig:
    cell_score_threshold: float
    num_workers: int


@dataclass
class ExtractConfig:
    edge_move_threshold: int
    block_size: List[int]
    num_workers: int


@dataclass
class SolveConfig:
    # TODO: is this the same as tracking parameters?
    cost_appear: float
    cost_disappear: float
    cost_split: float
    threshold_node_score: float
    weight_node_score: float
    threshold_edge_score: float
    weight_distance_cost: float
    weight_prediction_distance_cost: float
    block_size: List[int]
    context: List[int]
    num_workers: int
    from_scratch: bool


@dataclass
class EvaluateConfig:
    gt_db_name: str
    from_scratch: bool
