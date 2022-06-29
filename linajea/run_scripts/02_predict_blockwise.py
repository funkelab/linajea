from __future__ import absolute_import
import argparse
import logging
import time

from linajea.utils import (print_time,
                           getNextInferenceData)
from linajea.process_blockwise import predict_blockwise


logging.basicConfig(
        # wrong crop: switch
        level=logging.INFO,
        # level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint/iteration to predict')
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
    args = parser.parse_args()

    start_time = time.time()
    for inf_config in getNextInferenceData(args):
        predict_blockwise(inf_config)
    end_time = time.time()
    print_time(end_time - start_time)
