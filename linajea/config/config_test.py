from linajea import load_config
from linajea.config import (
        GeneralConfig,
        DataConfig,
        JobConfig,
        PredictConfig,
        ExtractConfig,
        SolveConfig,
        EvaluateConfig,
        )

if __name__ == "__main__":
    config_dict = load_config("sample_config.toml")
    general_config = GeneralConfig(**config_dict['general'])
    print(general_config)
    data_config = DataConfig(**config_dict['data'])
    print(data_config)
    job_config = JobConfig(**config_dict['job'])
    print(job_config)
    predict_config = PredictConfig(**config_dict['predict'])
    print(predict_config)
    extract_config = ExtractConfig(**config_dict['extract'])
    print(extract_config)
    solve_config = SolveConfig(**config_dict['solve'])
    print(solve_config)
    evaluate_config = EvaluateConfig(**config_dict['evaluate'])
    print(evaluate_config)
