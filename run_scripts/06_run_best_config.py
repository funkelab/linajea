"""Get best val and eval on test run script

Finds the parameters with the best result on the validation data
Solves and evaluates with those on the test data
"""
from __future__ import absolute_import
import argparse
import logging
import os
import sys
import time

import attr
import pandas as pd
import toml

from linajea.config import (SolveParametersConfig,
                            TrackingConfig)
import linajea.evaluation
from linajea.process_blockwise import solve_blockwise
from linajea.utils import (print_time,
                           getNextInferenceData)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to process')
    parser.add_argument('--swap_val_test', action="store_true",
                        help='swap validation and test data?')
    parser.add_argument('--sort_by', type=str, default="sum_errors",
                        help=('Which metric to use to select best '
                              'parameters/weights'))
    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)
    logging.basicConfig(
        level=config.general.logging,
        handlers=[
            logging.FileHandler('run.log', mode='a'),
            logging.StreamHandler(sys.stdout),
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    score_columns = ['fn_edges', 'identity_switches',
                     'fp_divisions', 'fn_divisions']
    if not config.general.sparse:
        score_columns = ['fp_edges'] + score_columns

    results = {}
    samples = set()
    args.validation = not args.swap_val_test
    for sample_idx, inf_config in enumerate(getNextInferenceData(
            args, is_evaluate=True)):
        sample = inf_config.inference_data.data_source.datafile.filename
        checkpoint = inf_config.inference_data.checkpoint
        cell_score_threshold = inf_config.inference_data.cell_score_threshold
        samples.add(sample)
        logger.debug(
            "getting results for:", sample, checkpoint, cell_score_threshold)
        res = linajea.evaluation.get_results_sorted(
            inf_config,
            filter_params={"val": True},
            score_columns=score_columns,
            sort_by=args.sort_by)

        res = res.assign(
            checkpoint=checkpoint).assign(
            cell_score_threshold=cell_score_threshold)
        results[(
            os.path.basename(sample), checkpoint, cell_score_threshold)] = \
                res.reset_index()
    args.validation = not args.validation

    results = pd.concat(list(results.values())).reset_index()
    del results['_id']
    del results['param_id']

    solve_params = attr.fields_dict(SolveParametersConfig)
    by = [
        "matching_threshold",
        "weight_node_score",
        "selection_constant",
        "track_cost",
        "weight_division",
        "division_constant",
        "weight_child",
        "weight_continuation",
        "weight_edge_score",
        "checkpoint",
        "cell_score_threshold"
    ]
    if "cell_state_key" in results:
        by.append("cell_state_key")

    results = results.groupby(by, dropna=False, as_index=False).agg(
        lambda x: -1
        if len(x) != len(samples)
        else sum(x)
        if (not isinstance(x.iloc[0], list) and
            not isinstance(x.iloc[0], dict) and
            not isinstance(x.iloc[0], str)
            )
        else x.iloc[0])

    results = results[results.sum_errors != -1]
    results.sort_values(args.sort_by, ascending=True, inplace=True)

    for k in solve_params.keys():
        if k == "tag" and k not in results.iloc[0]:
            solve_params[k] = None
            continue
        if k == "cell_state_key" and k not in results.iloc[0]:
            solve_params[k] = None
            continue
        solve_params[k] = results.iloc[0][k]
    solve_params['val'] = False

    config.path = os.path.join("tmp_configs", "config_{}.toml".format(
        time.time()))
    config_dict = attr.asdict(config)
    config_dict['solve']['parameters'] = [solve_params]
    config_dict['solve']['grid_search'] = False
    config_dict['solve']['random_search'] = False
    with open(config.path, 'w') as f:
        toml.dump(config_dict, f, encoder=toml.TomlNumpyEncoder())
    args.config = config.path

    start_time = time.time()
    for inf_config in getNextInferenceData(args, is_solve=True):
        solve_blockwise(inf_config)
    end_time = time.time()
    print_time(end_time - start_time)

    start_time = time.time()
    for inf_config in getNextInferenceData(args, is_evaluate=True):
        linajea.evaluation.evaluate_setup(inf_config)
    end_time = time.time()
    print_time(end_time - start_time)
