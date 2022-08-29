"""Evaluate run script

Loads the configuration and solves the ILP.
Expects data specified as [validate_data] and [test_data]
Automatically selects data; if db name not set, set automatically
based on data.
If weights/parameters search is supposed to be evaluated and run
separately from 04_solve.py/without run.py, disable grid_search and
random_search and supply --param_ids to select which parameter sets
stored in the database should be evaluated.
"""
import argparse
import logging
import sys
import time

from linajea.config import load_config
import linajea.evaluation
from linajea.utils import (print_time,
                           getNextInferenceData)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to process')
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
    parser.add_argument('--val_param_id', type=int, default=None,
                        help=('get test parameters from validation '
                              'parameters_id'))
    parser.add_argument('--param_id', default=None,
                        help='process parameters with parameters_id')
    parser.add_argument('--param_ids', default=None, nargs="+",
                        help='start/end range or list of eval parameters_ids')
    parser.add_argument('--param_list_idx', type=str, default=None,
                        help='only eval parameters[idx] in config')
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler('run.log', mode='a'),
            logging.StreamHandler(sys.stdout),
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    start_time = time.time()
    for inf_config in getNextInferenceData(args, is_evaluate=True):
        scores = linajea.evaluation.evaluate_setup(inf_config)
        if scores:
            logger.info("scores: %s", scores.get_short_report())
        else:
            logger.warning("unable to compute score for %s", inf_config)
    end_time = time.time()
    print_time(end_time - start_time)
