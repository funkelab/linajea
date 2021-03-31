import pandas
from linajea import CandidateDatabase
from linajea.tracking import TrackingParameters
import re
import logging

logger = logging.getLogger(__name__)


def get_sample_from_setup(setup):
    sample_int = int(re.search(re.compile(r"\d*"), setup).group())
    if sample_int < 100:
        return '140521'
    elif sample_int < 200:
        return '160328'
    elif sample_int < 300:
        return '120828'
    else:
        raise ValueError("Setup number must be < 300 to infer sample")


def get_result(
        setup,
        region,
        tracking_parameters,
        db_host,
        frames=None,
        sample=None,
        iteration='400000'):
    ''' Get the scores, statistics, and parameters for given
    setup, region, and parameters.
    Returns a dictionary containing the keys and values of the score
    object.

    tracking_parameters can be a dict or a TrackingParameters object'''
    if not sample:
        sample = get_sample_from_setup(setup)
    db_name = '_'.join(['linajea', sample, setup, region, iteration])
    candidate_db = CandidateDatabase(db_name, db_host, 'r')
    if isinstance(tracking_parameters, dict):
        tracking_parameters = TrackingParameters(**tracking_parameters)
    parameters_id = candidate_db.get_parameters_id(
            tracking_parameters,
            fail_if_not_exists=True)
    result = candidate_db.get_score(parameters_id, frames=frames)
    return result


def get_tgmm_results(
        region,
        db_host,
        sample,
        frames=None):
    if region is None:
        db_name = '_'.join(['linajea', sample, 'tgmm'])
    else:
        db_name = '_'.join(['linajea', sample, 'tgmm', region])
    candidate_db = CandidateDatabase(db_name, db_host, 'r')
    results = candidate_db.get_scores(frames=frames)
    if results is None or len(results) == 0:
        return None
    all_results = pandas.DataFrame(results)
    return all_results


def get_best_tgmm_result(
        region,
        db_host,
        sample,
        frames=None,
        score_columns=None,
        score_weights=None):
    if not score_columns:
        score_columns = ['fn_edges', 'identity_switches',
                         'fp_divisions', 'fn_divisions']
    if not score_weights:
        score_weights = [1.]*len(score_columns)
    results_df = get_tgmm_results(region, db_host, sample, frames=frames)
    if results_df is None:
        logger.warn("No TGMM results for region %s, sample %s, and frames %s"
                    % (region, sample, str(frames)))
        return None
    results_df['sum_errors'] = sum([results_df[col]*weight for col, weight
                                   in zip(score_columns, score_weights)])
    results_df.sort_values('sum_errors', inplace=True)
    best_result = results_df.iloc[0].to_dict()
    best_result['setup'] = 'TGMM'
    return best_result


def get_results(
        setup,
        region,
        db_host,
        sample=None,
        iteration='400000',
        frames=None,
        filter_params=None):
    ''' Gets the scores, statistics, and parameters for all
    grid search configurations run for the given setup and region.
    Returns a pandas dataframe with one row per configuration.'''
    if not sample:
        sample = get_sample_from_setup(setup)
    db_name = '_'.join(['linajea', sample, setup, region, iteration])
    candidate_db = CandidateDatabase(db_name, db_host, 'r')
    scores = candidate_db.get_scores(frames=frames, filters=filter_params)
    dataframe = pandas.DataFrame(scores)
    logger.debug("data types of dataframe columns: %s"
                 % str(dataframe.dtypes))
    if 'param_id' in dataframe:
        dataframe['_id'] = dataframe['param_id']
        dataframe.set_index('param_id', inplace=True)
    return dataframe


def get_best_result(setup, region, db_host,
                    sample=None,
                    iteration='400000',
                    frames=None,
                    filter_params=None,
                    score_columns=None,
                    score_weights=None):
    ''' Gets the best result for the given setup and region according to
    the sum of errors in score_columns, with optional weighting.

    Returns a dictionary'''
    if not score_columns:
        score_columns = ['fn_edges', 'identity_switches',
                         'fp_divisions', 'fn_divisions']
    if not score_weights:
        score_weights = [1.]*len(score_columns)
    results_df = get_results(setup, region, db_host,
                             frames=frames,
                             sample=sample, iteration=iteration,
                             filter_params=filter_params)
    results_df['sum_errors'] = sum([results_df[col]*weight for col, weight
                                   in zip(score_columns, score_weights)])
    results_df.sort_values('sum_errors', inplace=True)
    best_result = results_df.iloc[0].to_dict()
    for key, value in best_result.items():
        try:
            best_result[key] = value.item()
        except AttributeError:
            pass
    best_result['setup'] = setup
    return best_result


def get_best_result_per_setup(setups, region, db_host,
                              frames=None, sample=None, iteration='400000',
                              filter_params=None,
                              score_columns=None, score_weights=None):
    ''' Returns the best result for each setup in setups
    according to the sum of errors in score_columns, with optional weighting,
    sorted from best to worst (lowest to highest sum errors)'''
    best_results = []
    for setup in setups:
        best = get_best_result(setup, region, db_host,
                               frames=frames,
                               sample=sample, iteration=iteration,
                               filter_params=filter_params,
                               score_columns=score_columns,
                               score_weights=score_weights)
        best_results.append(best)

    best_df = pandas.DataFrame(best_results)
    best_df.sort_values('sum_errors', inplace=True)
    return best_df
