"""Extract Edges run script

Loads the configuration and computes edge candidates.
Expects data specified as [validate_data] and [test_data]
Automatically selects data; if db name not set, set automatically
based on data
"""
import argparse
import logging
import sys
import time

from linajea.config import load_config
from linajea.process_blockwise import extract_edges_blockwise
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
    for inf_config in getNextInferenceData(args):
        extract_edges_blockwise(inf_config)
    end_time = time.time()
    print_time(end_time - start_time)
