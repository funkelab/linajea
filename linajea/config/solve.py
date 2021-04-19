import attr
import itertools
import logging
import os
import random
from typing import List

from .data import DataROIConfig
from .job import JobConfig
from .utils import (ensure_cls,
                    ensure_cls_list)

logger = logging.getLogger(__name__)


def convert_solve_params_list():
    def converter(vals):
        if vals is None:
            return None
        if not isinstance(vals, list):
            vals = [vals]
        converted = []
        for val in vals:
            if isinstance(val, SolveParametersMinimalConfig):
                converted.append(val)
            elif isinstance(val, SolveParametersNonMinimalConfig):
                converted.append(val)
            else:
                if "track_cost" in val:
                    converted.append(SolveParametersMinimalConfig(**val))
                else:
                    converted.append(SolveParametersNonMinimalConfig(**val))
        return converted
    return converter


def convert_solve_search_params():
    def converter(vals):
        if vals is None:
            return None

        if isinstance(vals, SolveParametersMinimalSearchConfig) or \
           isinstance(vals, SolveParametersNonMinimalSearchConfig):
            return vals
        else:
            if "track_cost" in vals:
                return SolveParametersMinimalSearchConfig(**vals)
            else:
                return SolveParametersNonMinimalSearchConfig(**vals)
    return converter


@attr.s(kw_only=True)
class SolveParametersMinimalConfig:
    track_cost = attr.ib(type=float)
    weight_node_score = attr.ib(type=float)
    selection_constant = attr.ib(type=float)
    weight_division = attr.ib(type=float, default=0.0)
    division_constant = attr.ib(type=float, default=0.0)
    weight_child = attr.ib(type=float, default=0.0)
    weight_continuation = attr.ib(type=float, default=0.0)
    weight_edge_score = attr.ib(type=float)
    cell_cycle_key = attr.ib(type=str, default=None)
    block_size = attr.ib(type=List[int])
    context = attr.ib(type=List[int])
    # max_cell_move: currently use edge_move_threshold from extract
    max_cell_move = attr.ib(type=int, default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)

    def valid(self):
        return {key: val
                for key, val in attr.asdict(self).items()
                if val is not None}

    def query(self):
        params_dict_valid = self.valid()
        params_dict_none = {key: {"$exists": False}
                            for key, val in attr.asdict(self).items()
                            if val is None}
        query = {**params_dict_valid, **params_dict_none}
        return query


@attr.s(kw_only=True)
class SolveParametersMinimalSearchConfig:
    track_cost = attr.ib(type=List[float])
    weight_node_score = attr.ib(type=List[float])
    selection_constant = attr.ib(type=List[float])
    weight_division = attr.ib(type=List[float])
    division_constant = attr.ib(type=List[float])
    weight_child = attr.ib(type=List[float])
    weight_continuation = attr.ib(type=List[float])
    weight_edge_score = attr.ib(type=List[float])
    cell_cycle_key = attr.ib(type=str, default=None)
    block_size = attr.ib(type=List[List[int]])
    context = attr.ib(type=List[List[int]])
    # max_cell_move: currently use edge_move_threshold from extract
    max_cell_move = attr.ib(type=List[int], default=None)
    random_search = attr.ib(type=bool, default=False)
    num_random_configs = attr.ib(type=int, default=None)


@attr.s(kw_only=True)
class SolveParametersNonMinimalConfig:
    cost_appear = attr.ib(type=float)
    cost_disappear = attr.ib(type=float)
    cost_split = attr.ib(type=float)
    threshold_node_score = attr.ib(type=float)
    weight_node_score = attr.ib(type=float)
    threshold_edge_score = attr.ib(type=float)
    weight_prediction_distance_cost = attr.ib(type=float)
    use_cell_state = attr.ib(type=str, default=None)
    use_cell_cycle_indicator = attr.ib(type=str, default=False)
    threshold_split_score = attr.ib(type=float, default=None)
    threshold_is_normal_score = attr.ib(type=float, default=None)
    threshold_is_daughter_score = attr.ib(type=float, default=None)
    cost_daughter = attr.ib(type=float, default=None)
    cost_normal = attr.ib(type=float, default=None)
    prefix = attr.ib(type=str, default=None)
    block_size = attr.ib(type=List[int])
    context = attr.ib(type=List[int])
    # max_cell_move: currently use edge_move_threshold from extract
    max_cell_move = attr.ib(type=int, default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)

    def valid(self):
        return {key: val
                for key, val in attr.asdict(self).items()
                if val is not None}

    def query(self):
        params_dict_valid = self.valid()
        params_dict_none = {key: {"$exists": False}
                            for key, val in attr.asdict(self).items()
                            if val is None}
        query = {**params_dict_valid, **params_dict_none}
        return query


@attr.s(kw_only=True)
class SolveParametersNonMinimalSearchConfig:
    cost_appear = attr.ib(type=List[float])
    cost_disappear = attr.ib(type=List[float])
    cost_split = attr.ib(type=List[float])
    threshold_node_score = attr.ib(type=List[float])
    weight_node_score = attr.ib(type=List[float])
    threshold_edge_score = attr.ib(type=List[float])
    weight_prediction_distance_cost = attr.ib(type=List[float])
    use_cell_state = attr.ib(type=List[float], default=None)
    threshold_split_score = attr.ib(type=List[float], default=None)
    threshold_is_normal_score = attr.ib(type=List[float], default=None)
    threshold_is_daughter_score = attr.ib(type=List[float], default=None)
    cost_daughter = attr.ib(type=List[float], default=None)
    cost_normal = attr.ib(type=List[float], default=None)
    prefix = attr.ib(type=str, default=None)
    block_size = attr.ib(type=List[List[int]])
    context = attr.ib(type=List[List[int]])
    # max_cell_move: currently use edge_move_threshold from extract
    max_cell_move = attr.ib(type=List[int], default=None)
    random_search = attr.ib(type=bool, default=False)
    num_random_configs = attr.ib(type=int, default=None)

def write_solve_parameters_configs(parameters_search, non_minimal):
    params = {k:v
              for k,v in attr.asdict(parameters_search).items()
              if v is not None}
    del params['random_search']
    del params['num_random_configs']

    if parameters_search.random_search:
        search_configs = []
        assert parameters_search.num_random_configs is not None, \
            "set number_configs kwarg when using random search!"

        for _ in range(parameters_search.num_random_configs):
            conf = {}
            for k, v in params.items():
                if not isinstance(v, list):
                    conf[k] = v
                elif len(v) == 1:
                    conf[k] = v[0]
                elif isinstance(v[0], str):
                    rnd = random.choice(v)
                    if rnd == "":
                        rnd = None
                    conf[k] = rnd
                else:
                    assert len(v) == 2, \
                        "possible options per parameter for random search: " \
                        "single fixed value, upper and lower bound, " \
                        "set of string values ({})".format(v)
                    if isinstance(v[0], list):
                        idx = random.randrange(len(v[0]))
                        conf[k] = random.uniform(v[0][idx], v[1][idx])
                    else:
                        conf[k] = random.uniform(v[0], v[1])
            search_configs.append(conf)
    else:
        search_configs = [
            dict(zip(params.keys(), x))
            for x in itertools.product(*params.values())]

    configs = []
    for config_vals in search_configs:
        logger.debug("Config vals %s" % str(config_vals))
        if non_minimal:
            configs.append(SolveParametersNonMinimalConfig(
                **config_vals))  # type: ignore
        else:
            configs.append(SolveParametersMinimalConfig(
                **config_vals))  # type: ignore

    return configs


@attr.s(kw_only=True)
class SolveConfig:
    job = attr.ib(converter=ensure_cls(JobConfig))
    from_scratch = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=convert_solve_params_list(), default=None)
    parameters_search = attr.ib(converter=convert_solve_search_params(), default=None)
    non_minimal = attr.ib(type=bool, default=False)
    write_struct_svm = attr.ib(type=bool, default=False)
    check_node_close_to_roi = attr.ib(type=bool, default=True)
    add_node_density_constraints = attr.ib(type=bool, default=False)

    def __attrs_post_init__(self):
        assert self.parameters is not None or \
            self.parameters_search is not None, \
            "provide either solve parameters or grid/random search values " \
            "for solve parameters!"

        if self.parameters_search is not None:
            if self.parameters is not None:
                logger.warning("overwriting explicit solve parameters with "
                               "grid/random search parameters!")
            self.parameters = write_solve_parameters_configs(
                self.parameters_search, non_minimal=self.non_minimal)

        # block size and context must be the same for all parameters!
        block_size = self.parameters[0].block_size
        context = self.parameters[0].context
        for i in range(len(self.parameters)):
            assert block_size == self.parameters[i].block_size, \
                "%s not equal to %s" %\
                (block_size, self.parameters[i].block_size)
            assert context == self.parameters[i].context, \
                "%s not equal to %s" %\
                (context, self.parameters[i].context)
