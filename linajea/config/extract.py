from dataclasses import dataclass
from typing import List


@dataclass
class ExtractConfig:
    edge_move_threshold: int
    block_size: List[int]
    num_workers: int
