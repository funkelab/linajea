from .general import GeneralConfig
from .predict import PredictConfig
from .extract import ExtractConfig
from .evaluate import EvaluateConfig
from linajea import load_config


class LinajeaConfig:

    def __init__(self, config_file):
        config_dict = load_config(config_file)
        self.general = GeneralConfig(**config_dict['general'])
        self.predict = PredictConfig(**config_dict['predict'])
        self.extract = ExtractConfig(**config_dict['extract_edges'])
        self.evaluate = EvaluateConfig(**config_dict['evaluate'])
