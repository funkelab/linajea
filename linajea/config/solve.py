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

@attr.s(kw_only=True)
class SolveParametersConfig:
    track_cost = attr.ib(type=float)
    weight_node_score = attr.ib(type=float)
    selection_constant = attr.ib(type=float)
    weight_division = attr.ib(type=float)
    division_constant = attr.ib(type=float)
    weight_child = attr.ib(type=float)
    weight_continuation = attr.ib(type=float)
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
class SolveParametersSearchConfig:
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
    params = attr.asdict(parameters_search)
    del params['random_search']
    del params['num_random_configs']

    search_keys = list(params.keys())

    if parameters_search.random_search:
        search_configs = []
        assert parameters_search.num_random_configs is not None, \
            "set number_configs kwarg when using random search!"

        for _ in range(parameters_search.num_random_configs):
            conf = []
            for _, v in params.items():
                if not isinstance(v, list):
                    conf.append(v)
                elif len(v) == 1:
                    conf.append(v[0])
                elif isinstance(v[0], str):
                    conf.append(random.choice(v))
                else:
                    assert len(v) == 2, \
                        "possible options per parameter for random search: " \
                        "single fixed value, upper and lower bound, " \
                        "set of string values"
                    if isinstance(v[0], list):
                        idx = random.randrange(len(v[0]))
                        conf.append(random.uniform(v[0][idx], v[1][idx]))
                    else:
                        conf.append(random.uniform(v[0], v[1]))
            search_configs.append(conf)
    else:
        search_configs = itertools.product(*[params[key]
                                             for key in search_keys])

    configs = []
    for config_vals in search_configs:
        logger.debug("Config vals %s" % str(config_vals))
        if non_minimal:
            configs.append(SolveParametersNonMinimalConfig(
                **config_vals))  # type: ignore
        else:
            configs.append(SolveParametersConfig(
                **config_vals))  # type: ignore

    return configs


@attr.s(kw_only=True)
class SolveConfig:
    job = attr.ib(converter=ensure_cls(JobConfig))
    from_scratch = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=ensure_cls_list(
        SolveParametersConfig), default=None)
    parameters_search = attr.ib(converter=ensure_cls(
        SolveParametersSearchConfig), default=None)
    parameters_non_minimal = attr.ib(converter=ensure_cls_list(
        SolveParametersNonMinimalConfig), default=None)
    parameters_non_minimal_search = attr.ib(converter=ensure_cls(
        SolveParametersNonMinimalSearchConfig), default=None)
    non_minimal = attr.ib(type=bool, default=False)
    write_struct_svm = attr.ib(type=bool, default=False)
    check_node_close_to_roi = attr.ib(type=bool, default=True)
    add_node_density_constraints = attr.ib(type=bool, default=False)

    def __attrs_post_init__(self):
        assert self.parameters is not None or \
            self.parameters_search is not None or \
            self.parameters_non_minimal is not None or \
            self.parameters_non_minimal_search is not None, \
            "provide either solve parameters or grid/random search values " \
            "for solve parameters!"

        if self.parameters is not None or self.parameters_search is not None:
            assert not self.non_minimal, \
                "please set non_minimal to false when using minimal ilp"
        elif self.parameters_non_minimal is not None or \
             self.parameters_non_minimal_search is not None:
            assert self.non_minimal, \
                "please set non_minimal to true when using non minimal ilp"

        if self.parameters_search is not None:
            if self.parameters is not None:
                logger.warning("overwriting explicit solve parameters with "
                               "grid/random search parameters!")
            self.parameters = write_solve_parameters_configs(
                self.parameters_search, non_minimal=False)
        elif self.parameters_non_minimal_search is not None:
            if self.parameters_non_minimal is not None:
                logger.warning("overwriting explicit solve parameters with "
                               "grid/random search parameters!")
            self.parameters = write_solve_parameters_configs(
                self.parameters_non_minimal_search, non_minimal=True)
        elif self.parameters_non_minimal is not None:
            assert self.parameters is None, \
                "overwriting minimal ilp parameters with non-minimal ilp ones"
            self.parameters = self.parameters_non_minimal

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
