import argparse
import logging
import time

from linajea.utils import (print_time,
                           getNextInferenceData)
import linajea.evaluation

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
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
                        help='get test parameters from validation parameters_id')
    parser.add_argument('--param_id', default=None,
                        help='process parameters with parameters_id')
    parser.add_argument('--param_list_idx', type=str, default=None,
                        help='only eval parameters[idx] in config')
    args = parser.parse_args()

    start_time = time.time()
    for inf_config in getNextInferenceData(args, is_evaluate=True):
        linajea.evaluation.evaluate_setup(inf_config)
    end_time = time.time()
    print_time(end_time - start_time)
