"""Provides a function to pretty print a time duration
"""
import logging

logger = logging.getLogger(__name__)


def print_time(seconds):
    ''' A function to pretty print a number of seconds as
    days, hours, minutes, seconds
    '''
    logger.debug("Time spent in seconds: {}".format(seconds))
    seconds = int(seconds)
    days = seconds // 86400
    seconds -= 86400 * days
    hours = seconds // 3600
    seconds -= 3600 * hours
    minutes = seconds // 60
    seconds -= 60 * minutes
    logger.info("Time spent: %d days %d hours %d minutes %d seconds"
                % (days, hours, minutes, seconds))
