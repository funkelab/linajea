"""Provides a script to write (grid) search parameters as list to toml file
"""
import argparse
import itertools
import logging
import random

import toml

from linajea.config import load_config

logger = logging.getLogger(__name__)


def write_search_configs(config, random_search, output_file, num_configs=None):
    """Create list of ILP weights sets based on configuration

    Args
    ----
    parameters_search: dict
        Parameter search object that is used to create list of
        individual sets of parameters
    random_search: bool
        Do grid search or random search
    output_file: str
        Write configurations to this file
    num_configs: int
        How many configurations to create; if None random has to be false,
        then all possible grid configurations will be created, otherwise
        a random subset of them.

    Returns
    -------
    List of ILP weights configurations


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
    params = {k:v
              for k,v in config.items()
              if v is not None}
    params.pop('num_configs', None)

    if random_search:
        search_configs = []
        assert num_configs is not None, \
            "set num_configs kwarg when using random search!"

        for _ in range(num_configs):
            conf = {}
            for k, v in params.items():
                if not isinstance(v, list):
                    value = v
                elif len(v) == 1:
                    value = v[0]
                elif isinstance(v[0], str) or len(v) > 2:
                    value = random.choice(v)
                elif len(v) == 2 and isinstance(v[0], list) and isinstance(v[1], list) and \
                     isinstance(v[0][0], str) and isinstance(v[1][0], str):
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
        search_configs = [
            dict(zip(params.keys(), x))
            for x in itertools.product(*params.values())]

        if num_configs and num_configs <= len(search_configs):
            random.shuffle(search_configs)
            search_configs = search_configs[:num_configs]

    search_configs = {"parameters": search_configs}
    with open(output_file, 'w') as f:
        print(search_configs)
        toml.dump(search_configs, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='sample config file')
    parser.add_argument('--output', type=str, required=True,
                        help='output file')
    parser.add_argument('--num_configs', type=int, default=None,
                        help='how many configurations to create')
    parser.add_argument('--random', action="store_true",
                        help='do random search (as opposed to grid search)')
    args = parser.parse_args()

    config = load_config(args.config)
    write_search_configs(config, args.random, args.output,
                         num_configs=args.num_configs)
