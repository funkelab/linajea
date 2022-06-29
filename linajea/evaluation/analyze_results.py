import logging

import pandas as pd

from linajea.utils import CandidateDatabase

logger = logging.getLogger(__name__)


def get_results_sorted(config,
                       filter_params=None,
                       score_columns=None,
                       score_weights=None,
                       sort_by="sum_errors"):
    """Get sorted results based on config

    Args
    ----
    config: TrackingConfig
        Config object used to determine database name and host
        and evaluation parameters
    filter_params: dict
        Has to be a valid mongodb query, used to filter results
    score_columns: list of str
        Which columns should count towards the sum of errors
    score_weights: list of float
        Option to use non-uniform weights per type of error
    sort_by: str
        Sort by which column/type of error (by default: sum)

    Returns
    -------
    pandas.DataFrame
        Sorted results stored in pandas.DataFrame object
    """
    db_name = config.inference_data.data_source.db_name

    return get_results_sorted_db(db_name,
                                 config.general.db_host,
                                 filter_params=filter_params,
                                 eval_params=config.evaluate.parameters,
                                 score_columns=score_columns,
                                 score_weights=score_weights,
                                 sort_by=sort_by)


def get_best_result_config(config,
                           filter_params=None,
                           score_columns=None,
                           score_weights=None):
    """Get best result based on config

    Args
    ----
    config: TrackingConfig
        Config object used to determine database name and host
        and evaluation parameters
    filter_params: dict
        Has to be a valid mongodb query, used to filter results
    score_columns: list of str
        Which columns should count towards the sum of errors
    score_weights: list of float
        Option to use non-uniform weights per type of error
    sort_by: str
        Sort by which column/type of error (by default: sum)

    Returns
    -------
    dict
        Get best result stored in dict
        Includes, parameters used, parameter id and scores/errors
    """
    results_df = get_results_sorted(config,
                                    filter_params=filter_params,
                                    score_columns=score_columns,
                                    score_weights=score_weights)
    best_result = results_df.iloc[0].to_dict()
    for key, value in best_result.items():
        try:
            best_result[key] = value.item()
        except AttributeError:
            pass
    return best_result


def get_results_sorted_db(db_name,
                          db_host,
                          filter_params=None,
                          eval_params=None,
                          score_columns=None,
                          score_weights=None,
                          sort_by="sum_errors"):
    """Get sorted results from given database

    Args
    ----
    db_name: str
        Which database to use
    db_host: str
        Which database connection/host to use
    filter_params: dict
        Has to be a valid mongodb query, used to filter results
    eval_params: EvaluateParametersConfig
        Evaluation parameters config object, used to filter results
    score_columns: list of str
        Which columns should count towards the sum of errors
    score_weights: list of float
        Option to use non-uniform weights per type of error
    sort_by: str
        Sort by which column/type of error (by default: sum)

    Returns
    -------
    pandas.DataFrame
        Sorted results stored in pandas.DataFrame object
    """
    if not score_columns:
        score_columns = ['fn_edges', 'identity_switches',
                         'fp_divisions', 'fn_divisions']
    if not score_weights:
        score_weights = [1.]*len(score_columns)

    logger.info("Getting results in db: %s", db_name)
    candidate_db = CandidateDatabase(db_name, db_host, 'r')
    scores = candidate_db.get_scores(filters=filter_params,
                                     eval_params=eval_params)


    if len(scores) == 0:
        raise RuntimeError("no scores found!")

    results_df = pd.DataFrame(scores)
    if 'param_id' in results_df:
        results_df['_id'] = results_df['param_id']
        results_df.set_index('param_id', inplace=True)

    results_df['sum_errors'] = sum([results_df[col]*weight for col, weight
                                   in zip(score_columns, score_weights)])
    results_df['sum_divs'] = sum(
        [results_df[col]*weight for col, weight
         in zip(score_columns[-2:], score_weights[-2:])])
    results_df = results_df.astype({"sum_errors": int, "sum_divs": int})
    ascending = True
    if sort_by == "matched_edges":
        ascending = False
    results_df.sort_values(sort_by, ascending=ascending, inplace=True)
    return results_df


def get_result_id(
        config,
        parameters_id):
    ''' Get the scores, statistics, and parameters for given args

    Args
    ----
    config: TrackingConfig
        Configuration object, used to select database
    parameters_id: int
        Parameter ID, used to select solution within database

    Returns
    -------
    dict
        a dictionary containing the keys and values of the score object.
    '''
    db_name = config.inference_data.data_source.db_name
    candidate_db = CandidateDatabase(db_name, config.general.db_host, 'r')

    result = candidate_db.get_score(parameters_id,
                                    eval_params=config.evaluate.parameters)
    return result


def get_result_params(
        config,
        parameters):
    ''' Get the scores, statistics, and parameters for given args

    Args
    ----
    config: TrackingConfig
        Configuration object, used to select database
    parameters: int
        Set of parameters, used to select solution within database

    Returns
    -------
    dict
        a dictionary containing the keys and values of the score object.
    '''
    db_name = config.inference_data.data_source.db_name
    candidate_db = CandidateDatabase(db_name, config.general.db_host, 'r')
    if config.evaluate.parameters.roi is None:
        config.evaluate.parameters.roi = config.inference_data.data_source.roi

    result = candidate_db.get_score(
        candidate_db.get_parameters_id_round(
            parameters,
            fail_if_not_exists=True),
        eval_params=config.evaluate.parameters)
    return result
