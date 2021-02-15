import attr

from linajea import load_config
from linajea.config import (
    LinajeaConfig,
    TrackingConfig,
    CellCycleConfig,
    GeneralConfig,
    DataConfig,
    JobConfig,
    PredictConfig,
    ExtractConfig,
    SolveConfig,
    EvaluateConfig,
    VGGConfig,
    ResNetConfig,
    EfficientNetConfig,
)

if __name__ == "__main__":
    # parts of config
    config_dict = load_config("sample_config_tracking.toml")
    general_config = GeneralConfig(**config_dict['general'])
    print(general_config)
    data_config = DataConfig(**config_dict['train']['data'])
    print(data_config)
    job_config = JobConfig(**config_dict['train']['job'])
    print(job_config)

    # tracking parts of config
    predict_config = PredictConfig(**config_dict['predict'])
    print(predict_config)
    extract_config = ExtractConfig(**config_dict['extract'])
    print(extract_config)
    solve_config = SolveConfig(**config_dict['solve'])
    print(solve_config)
    evaluate_config = EvaluateConfig(**config_dict['evaluate'])
    print(evaluate_config)


    # # cell cycle parts of config
    # config_dict = load_config("sample_config_cellcycle.toml")
    # vgg_config = VGGConfig(**config_dict['model'])
    # print(evaluate_config)


    # complete configs
    tracking_config = TrackingConfig(path="sample_config_tracking.toml", **config_dict)
    print(tracking_config)

    tracking_config = TrackingConfig.from_file("sample_config_tracking.toml")
    print(tracking_config)

    config_dict = load_config("sample_config_cellcycle.toml")
    cell_cycle_config = CellCycleConfig(path="sample_config_cellcycle.toml", **config_dict)
    print(cell_cycle_config)

    print(attr.asdict(cell_cycle_config))
