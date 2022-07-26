"""Configuration used to define parameters for ILP solver
"""

import itertools
import logging
import random
from typing import List, Tuple

import attr

from .data import DataROIConfig
from .job import JobConfig
from .utils import (ensure_cls,
                    ensure_cls_list)

logger = logging.getLogger(__name__)


@attr.s(kw_only=True)
class SolveParametersConfig:
    """Defines a set of ILP hyperparameters

    Attributes
    ----------
    track_cost, weight_node_score, selection_constant, weight_division,
    division_constant, weight_child, weight_continuation,
    weight_edge_score: float
        main ILP hyperparameters
    cell_state_key: str
        key defining which cell state classifier to use
    block_size: list of int
        ILP is solved in blocks, defines size of each block
    context: list of int
        Size of context by which block is grown, to ensure consistent
        solution along borders
    max_cell_move: int
        How far a cell can move in one frame, cells closer than this
        value to the border do not have to pay certain costs
        (by default edge_move_threshold from extract is used)
    roi: DataROI
        Size of the data sample that is being "solved"
    feature_func: str
        Optional function that is applied to node and edge scores
        before incorporating them into the cost function.
        One of ["noop", "log", "square"]
    val: bool
        Is this set of parameters part of the validation
        parameter search or does it represent a test result?
        (if database is used once for testing and once for validation
        as part of cross-validation)
    tag: str
        To automatically tag e.g. ssvm/greedy solutions
    """
    track_cost = attr.ib(type=float)
    weight_node_score = attr.ib(type=float)
    selection_constant = attr.ib(type=float)
    weight_division = attr.ib(type=float, default=0.0)
    division_constant = attr.ib(type=float, default=1.0)
    weight_child = attr.ib(type=float, default=0.0)
    weight_continuation = attr.ib(type=float, default=0.0)
    weight_edge_score = attr.ib(type=float)
    cell_state_key = attr.ib(type=str, default=None)
    block_size = attr.ib(type=Tuple[int, int, int, int])
    context = attr.ib(type=Tuple[int, int, int, int])
    max_cell_move = attr.ib(type=int, default=None)
    roi = attr.ib(converter=ensure_cls(DataROIConfig), default=None)
    feature_func = attr.ib(type=str, default="noop")
    val = attr.ib(type=bool, default=False)
    tag = attr.ib(type=str, default=None)

    def valid(self):
        """Get all valid attributes

        Returns
        -------
            Dict with all parameters that are not None
        """
        return {key: val
                for key, val in attr.asdict(self).items()
                if val is not None}

    def query(self):
        """Get attributes for (querying database)

        Returns
        -------
            Dict with all valid parameters and all invalid parameters
            set to "exist=False"
        """
        params_dict_valid = self.valid()
        params_dict_none = {key: {"$exists": False}
                            for key, val in attr.asdict(self).items()
                            if val is None}
        query = {**params_dict_valid, **params_dict_none}
        return query


@attr.s(kw_only=True)
class SolveParametersSearchConfig:
    """Defines ranges/sets of ILP hyperparameters for a parameter search

    Can be used for both random search (search by selecting random
    values within given range) and grid search (search by
    cartesian product of given values per parameter)

    Notes
    -----
    For description of main attributes see SolveParametersConfig

    Attributes
    ----------
    num_configs: int
        How many sets of parameters to check.
        For random search: select this many sets of random values
        For grid search: shuffle cartesian product of parameters and
        take the num_configs first ones
    """
    track_cost = attr.ib(type=List[float])
    weight_node_score = attr.ib(type=List[float])
    selection_constant = attr.ib(type=List[float])
    weight_division = attr.ib(type=List[float], default=None)
    division_constant = attr.ib(type=List[float])
    weight_child = attr.ib(type=List[float], default=None)
    weight_continuation = attr.ib(type=List[float], default=None)
    weight_edge_score = attr.ib(type=List[float])
    cell_state_key = attr.ib(type=str, default=None)
    block_size = attr.ib(type=List[List[int]])
    context = attr.ib(type=List[List[int]])
    max_cell_move = attr.ib(type=List[int], default=None)
    feature_func = attr.ib(type=List[str], default=["noop"])
    val = attr.ib(type=List[bool], default=[True])
    num_configs = attr.ib(type=int, default=None)


def write_solve_parameters_configs(parameters_search, grid):
    """Create list of ILP hyperparameter sets based on configuration

    Args
    ----
    parameters_search: SolveParametersSearchConfig
        Parameter search object that is used to create list of
        individual sets of parameters
    grid: bool
        Do grid search or random search

    Returns
    -------
    List of ILP hyperparameter configurations


    Notes
    -----
    Random search:
        How random values are determined depends on respective type of
        values, number of combinations determined by num_configs
        Not a list or list with single value:
            interpreted as single value that is selected
        List of str or more than two values:
            interpreted as discrete options
        List of two lists:
            Sample outer list discretely. If inner list contains
            strings, sample discretely again; if inner list contains
            numbers, sample uniformly from range
        List of two numbers:
            Sample uniformly from range
    Grid search:
        Perform some type cleanup and compute cartesian product with
        itertools.product. If num_configs is set, shuffle list and take
        the num_configs first ones.
    """
    params = {k: v
              for k, v in attr.asdict(parameters_search).items()
              if v is not None}
    params.pop('num_configs', None)

    if not grid:
        search_configs = []
        assert parameters_search.num_configs is not None, \
            "set num_configs kwarg when using random search!"

        for _ in range(parameters_search.num_configs):
            conf = {}
            for k, v in params.items():
                if not isinstance(v, list):
                    value = v
                elif len(v) == 1:
                    value = v[0]
                elif isinstance(v[0], str) or len(v) > 2:
                    value = random.choice(v)
                elif (len(v) == 2 and isinstance(v[0], list) and
                      isinstance(v[1], list) and
                      isinstance(v[0][0], str) and isinstance(v[1][0], str)):
                    subset = random.choice(v)
                    value = random.choice(subset)
                else:
                    assert len(v) == 2, \
                        "possible options per parameter for random search: " \
                        "single fixed value, upper and lower bound, " \
                        "set of string values ({})".format(v)
                    if isinstance(v[0], list):
                        idx = random.randrange(len(v[0]))
                        value = random.uniform(v[0][idx], v[1][idx])
                    else:
                        value = random.uniform(v[0], v[1])
                if value == "":
                    value = None
                conf[k] = value
            search_configs.append(conf)
    else:
        if params.get('cell_state_key') == '':
            params['cell_state_key'] = None
        elif (isinstance(params.get('cell_state_key'), list) and
              '' in params['cell_state_key']):
            params['cell_state_key'] = [k if k != '' else None
                                        for k in params['cell_state_key']]
        search_configs = [
            dict(zip(params.keys(), x))
            for x in itertools.product(*params.values())]

        if parameters_search.num_configs:
            random.shuffle(search_configs)
            search_configs = search_configs[:parameters_search.num_configs]

    configs = []
    for config_vals in search_configs:
        logger.debug("Config vals %s" % str(config_vals))
        configs.append(SolveParametersConfig(
            **config_vals))  # type: ignore

    return configs


@attr.s(kw_only=True)
class SolveConfig:
    """Defines configuration for ILP solving step

    Attributes
    ----------
    job: JobConfig
        HPC cluster parameters, default constructed (executed locally)
        if not supplied
    from_scratch: bool
        If solution should be recomputed if it already exists
    parameters: SolveParametersConfig
        Fixed set of ILP parameters
    parameters_search_grid
    parameters_search_random: SolveParametersSearchConfig
        Ranges/sets per ILP parameter to create parameter search
    greedy: bool
        Do not use ILP for solving, greedy nearest neighbor tracking
    check_node_close_to_roi: bool
        If set, nodes close to roi border do not pay certain costs
        (as they can move outside of the field of view)
    timeout: int
        Time the solver has to find a solution
    clip_low_score: float
        Discard nodes with score lower than this value;
        Only useful if lower than threshold used during prediction
    grid_search, random_search: bool
        If grid and/or random search over ILP parameters should be
        performed
    solver_type: str
        Select preset type of Solver (set of constraints, indicators and
        cost functions), if None those have to be defined by calling
        function. Current options: `basic` and `cell_state`

    Notes
    -----
    The post init attrs function handles all the parameter setup, e.g.
    creating the parameter search configurations on the fly, if and
    which kind of search to perform etc.
    """
    job = attr.ib(converter=ensure_cls(JobConfig),
                  default=attr.Factory(JobConfig))
    from_scratch = attr.ib(type=bool, default=False)
    parameters = attr.ib(converter=ensure_cls_list(SolveParametersConfig),
                         default=None)
    parameters_search_grid = attr.ib(
        converter=ensure_cls(SolveParametersSearchConfig), default=None)
    parameters_search_random = attr.ib(
        converter=ensure_cls(SolveParametersSearchConfig), default=None)
    greedy = attr.ib(type=bool, default=False)
    write_struct_svm = attr.ib(type=str, default=None)
    check_node_close_to_roi = attr.ib(type=bool, default=True)
    timeout = attr.ib(type=int, default=0)
    clip_low_score = attr.ib(type=float, default=None)
    grid_search = attr.ib(type=bool, default=False)
    random_search = attr.ib(type=bool, default=False)
    solver_type = attr.ib(type=str, default=None,
                          validator=attr.validators.optional(
                              attr.validators.in_([
                                  "basic",
                                  "cell_state"])))

    def __attrs_post_init__(self):

        if self.grid_search or self.random_search:
            assert self.grid_search != self.random_search, \
                "choose either grid or random search!"
            assert self.parameters is None, \
                ("overwriting explicit solve parameters with grid/random "
                 "search parameters not supported. For search please either "
                 "(1) precompute search parameters (e.g. using "
                 "write_config_files.py) and set solve.parameters to point to "
                 "resulting file or (2) let search parameters be created "
                 "automatically by setting solve.grid/random_search to true "
                 "(only supported when using the getNextInferenceData "
                 "facility to loop over data samples)")
            if self.parameters is not None:
                logger.warning("overwriting explicit solve parameters with "
                               "grid/random search parameters!")
            if self.grid_search:
                assert self.parameters_search_grid is not None, \
                    "provide grid search values for solve parameters " \
                    "if grid search activated"
                parameters_search = self.parameters_search_grid
            else:
                assert self.parameters_search_random is not None, \
                    "provide random search values for solve parameters " \
                    "if random search activated"
                parameters_search = self.parameters_search_random
            self.parameters = write_solve_parameters_configs(
                parameters_search, grid=self.grid_search)

        if self.greedy:
            config_vals = {
                "weight_node_score": 0,
                "selection_constant": 0,
                "track_cost": 0,
                "weight_division": 0,
                "division_constant": 0,
                "weight_child": 0,
                "weight_continuation": 0,
                "weight_edge_score":  0,
                "block_size": [-1, -1, -1, -1],
                "context": [-1, -1, -1, -1],
                "max_cell_move": -1,
                "tag": "greedy",
            }
            self.parameters = [SolveParametersConfig(**config_vals)]

        if self.parameters is not None:
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
