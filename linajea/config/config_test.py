import attr

from linajea import load_config
from linajea.config import (
    TrackingConfig,
    CellCycleConfig,
    GeneralConfig,
    DataFileConfig,
    JobConfig,
    PredictTrackingConfig,
    ExtractConfig,
    SolveConfig,
    EvaluateTrackingConfig,
)

if __name__ == "__main__":
    # parts of config
    config_dict = load_config("sample_config_tracking.toml")
    general_config = GeneralConfig(**config_dict['general']) # type: ignore
    print(general_config)
    # data_config = DataFileConfig(**config_dict['train']['data']) # type: ignore
    # print(data_config)
    job_config = JobConfig(**config_dict['train']['job']) # type: ignore
    print(job_config)

    # tracking parts of config
    predict_config = PredictTrackingConfig(**config_dict['predict']) # type: ignore
    print(predict_config)
    extract_config = ExtractConfig(**config_dict['extract']) # type: ignore
    print(extract_config)
    solve_config = SolveConfig(**config_dict['solve']) # type: ignore
    print(solve_config)
    evaluate_config = EvaluateTrackingConfig(**config_dict['evaluate']) # type: ignore
    print(evaluate_config)


    # complete configs
    tracking_config = TrackingConfig(path="sample_config_tracking.toml", **config_dict) # type: ignore
    print(tracking_config)

    tracking_config = TrackingConfig.from_file("sample_config_tracking.toml")
    print(tracking_config)

    # config_dict = load_config("sample_config_cellcycle.toml")
    cell_cycle_config = CellCycleConfig.from_file("sample_config_cellcycle.toml") # type: ignore
    print(cell_cycle_config)

    # print(attr.asdict(cell_cycle_config))
