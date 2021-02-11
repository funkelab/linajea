from dataclasses import dataclass
from typing import List


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
