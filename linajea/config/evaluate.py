from dataclasses import dataclass


@dataclass
class EvaluateConfig:
    gt_db_name: str
    from_scratch: bool
    matching_threshold: int
