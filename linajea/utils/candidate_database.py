"""Provides a class wrapping a mongodb database

The database is used to store object and edge candidates, sets of
parameters that have been used to compute tracks from these and the
associated solutions and evaluation.
"""
import logging
import pymongo
import time

import numpy as np
from daisy.persistence import MongoDbGraphProvider
from daisy import Coordinate, Roi

logger = logging.getLogger(__name__)


class CandidateDatabase(MongoDbGraphProvider):
    '''Wrapper for daisy mongo graph that allows storing
    ILP parameter sets and scores, and easy retrieval of
    selected linage graphs.

    Arguments:

        db_name (``string``):

            The name of the MongoDB database.

        db_host (``string``):

            The URL of the MongoDB host.

        mode (``string``, optional):

            One of ``r``, ``r+``, or ``w``. Defaults to ``r+``. ``w`` drops the
            node, edge, meta, and parameters collections.

        total_roi (``bool``, optional):

            Gets written into graph metadata if provided.

        endpoint_names (``list`` or ``tuple`` with two elements, optional):

            What keys to use for the start and end of an edge. Default is
            ['source', 'target']

        parameters_id (``int``, optional):

           If provided, sets the parameters_id so that get_selected_graph
           will retrieve the results of solving with the given parameters
           and reset_selection will remove results of given parameters

    '''
    def __init__(
            self,
            db_name,
            db_host,
            mode='r',
            total_roi=None,
            endpoint_names=['source', 'target'],
            parameters_id=None):

        super().__init__(
                db_name,
                host=db_host,
                mode=mode,
                total_roi=total_roi,
                directed=True,
                position_attribute=['t', 'z', 'y', 'x'],
                endpoint_names=endpoint_names
                )
        if mode == 'w':
            try:
                self._MongoDbGraphProvider__connect()
                self._MongoDbGraphProvider__open_db()
                params_collection = self.database['parameters']
                params_collection.drop()
            finally:
                self._MongoDbGraphProvider__disconnect()
        self.parameters_id = None
        self.selected_key = None
        if parameters_id is not None:
            self.set_parameters_id(parameters_id)

    def get_selected_graph(self, roi, edge_attrs=None):
        '''Gets the edges selected by the candidate database's parameters_id
        within roi and all connected nodes. Ignores attribute keys on edges
        other than selected key of parameters_id to speed up retrieval and
        manipulation of resulting networx graph.
        '''
        if edge_attrs is None:
            edge_attrs = []
        edge_attrs.append(self.selected_key)

        assert self.selected_key is not None,\
            "No selected key provided, cannot get selected graph"
        subgraph = self.get_graph(
                roi,
                edges_filter={self.selected_key: True},
                edge_attrs=edge_attrs)
        unattached_nodes = [node for node in subgraph.nodes()
                            if subgraph.degree(node) == 0]
        subgraph.remove_nodes_from(unattached_nodes)
        return subgraph

    def reset_selection(self, roi=None, parameter_ids=None):
        ''' Removes all selections for self.parameters_id from mongodb
        edges collection
        '''
        if self.parameters_id is None and parameter_ids is None:
            logger.warn("No parameters id stored or provided:"
                        " cannot reset selection")
            return

        if roi:
            nodes = self.read_nodes(roi)
            node_ids = list([int(np.int64(n['id'])) for n in nodes])
            query = {self.endpoint_names[0]: {'$in': node_ids}}
        else:
            query = {}

        if not parameter_ids:
            parameter_ids = [self.parameters_id]
        logger.info("Resetting solution for parameter ids %s",
                    parameter_ids)
        start_time = time.time()
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        edge_coll = self.database['edges']
        update = {}
        for _id in parameter_ids:
            update['selected_' + str(_id)] = ""
        edge_coll.update_many(query, {'$unset': update})
        if roi is None:
            for _id in parameter_ids:
                daisy_coll_name = 'solve_' + str(_id) + '_daisy'
                self.database.drop_collection(daisy_coll_name)
        logger.info("Resetting soln for parameter_ids %s in roi %s"
                    " took %d seconds",
                    parameter_ids, roi, time.time() - start_time)

    def get_parameters_id(
            self,
            parameters,
            fail_if_not_exists=False):
        '''Get id for parameter set from mongo collection.
        If fail_if_not_exists, fail if the parameter set isn't already there.
        The default is to assign a new id and write it to the collection.
        '''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        params_id = None
        try:
            params_collection = self.database['parameters']
            query = parameters.query()
            del query['roi']
            logger.info("Querying ID for parameters %s", query)
            cnt = params_collection.count_documents(query)
            if fail_if_not_exists:
                assert cnt > 0, "Did not find id for parameters %s in %s"\
                    " and fail_if_not_exists set to True" % (
                        query, self.db_name)
            assert cnt <= 1, RuntimeError("multiple documents found in db"
                                          " for these parameters: %s: %s",
                                          query,
                                          list(params_collection.find(query)))
            if cnt == 1:
                find_result = params_collection.find_one(query)
                logger.info("Parameters %s already in collection with id %d"
                            % (query, find_result['_id']))
                params_id = find_result['_id']
            else:
                params_id = self.insert_with_next_id(parameters.valid(),
                                                     params_collection)
                logger.info("Parameters %s not yet in collection,"
                            " adding with id %d",
                            query, params_id)
        finally:
            self._MongoDbGraphProvider__disconnect()

        return int(params_id)

    def get_parameters_id_round(
            self,
            parameters,
            fail_if_not_exists=False):
        '''Get id for parameter set from mongo collection.
        If fail_if_not_exists, fail if the parameter set isn't already there.
        The default is to assign a new id and write it to the collection.
        If parameters are read from a text file there might be some floating
        point inaccuracies compared to values stored in the database.
        '''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        params_id = None
        try:
            params_collection = self.database['parameters']
            query = parameters.query()
            del query['roi']
            logger.info("Querying ID for parameters %s", query)
            entries = []
            params = list(query.keys())
            for entry in params_collection.find({}):
                if "context" not in entry:
                    continue
                for param in params:
                    if isinstance(query[param], float) or \
                       (isinstance(query[param], int) and
                        not isinstance(query[param], bool)):
                        if round(entry[param], 10) != round(query[param], 10):
                            break
                    elif isinstance(query[param], dict):
                        if param == 'roi' and \
                           (entry['roi']['offset'] != query['roi']['offset'] or \
                           entry['roi']['shape'] != query['roi']['shape']):
                            break
                        elif param == 'cell_cycle_key':
                            if '$exists' in query['cell_cycle_key'] and \
                               query['cell_cycle_key']['$exists'] == False and \
                               'cell_cycle_key' in entry:
                                break
                    elif isinstance(query[param], str):
                        if param not in entry or \
                           entry[param] != query[param]:
                            break
                    elif isinstance(query[param], list):
                        if entry[param] != query[param]:
                            break
                    elif isinstance(query[param], bool):
                        if param not in entry or \
                           entry[param] != query[param]:
                            break
                else:
                    entries.append(entry)
            cnt = len(entries)
            if fail_if_not_exists:
                assert cnt > 0, "Did not find id for parameters %s in %s"\
                    " and fail_if_not_exists set to True" % (
                        query, self.db_name)
            assert cnt <= 1, RuntimeError("multiple documents found in db"
                                          " for these parameters: %s: %s",
                                          query,
                                          entries)
            if cnt == 1:
                find_result = entries[0]
                logger.info("Parameters %s already in collection with id %d"
                            % (query, find_result['_id']))
                params_id = find_result['_id']
            else:
                params_id = self.insert_with_next_id(parameters.valid(),
                                                     params_collection)
                logger.info("Parameters %s not yet in collection,"
                            " adding with id %d",
                            query, params_id)
        finally:
            self._MongoDbGraphProvider__disconnect()

        return int(params_id)

    def insert_with_next_id(self, document, collection):
        '''Inserts a new set of parameters into the database, assigning
        the next sequential int id
        '''
        count_coll = self.database['parameters']

        while True:
            max_id = count_coll.find_one_and_update(
                    {'_id': 'parameters'},
                    {'$inc': {'id': 1}},
                    projection={'_id': 0},
                    upsert=True)
            if max_id:
                document['_id'] = max_id['id'] + 1
            else:
                document['_id'] = 1
            try:
                collection.insert_one(document)
            except pymongo.errors.DuplicateKeyError:
                # another worker beat you to this key - try again
                logger.info("Key %s already exists in parameter collection."
                            " Trying again" % document['_id'])
                continue
            break
        return document['_id']

    def get_parameters(self, params_id):
        '''Gets the parameters associated with the given id.
        Returns null if there are no parameters with the given id'''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        try:
            params_collection = self.database['parameters']
            params = params_collection.find_one({'_id': params_id})
            if params:
                del params['_id']
        finally:
            self._MongoDbGraphProvider__disconnect()
        return params

    def get_parameters_many(self, params_ids):
        '''Gets the parameters associated with the given ids.
        Returns None per id if there are no parameters with the given id'''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        try:
            params_collection = self.database['parameters']
            params_sets = []
            for pid in params_ids:
                params = params_collection.find_one({'_id': pid})
                if params:
                    del params['_id']
                params_sets.append(params)
        finally:
            self._MongoDbGraphProvider__disconnect()
        return params_sets

    def set_parameters_id(self, parameters_id):
        '''Sets the parameters_id and selected_key for the CandidateDatabase,
        so that you can use reset_selection and/or get_selected_graph'''
        self.parameters_id = int(parameters_id)
        self.selected_key = 'selected_' + str(self.parameters_id)
        logger.debug("Set selected_key to %s" % self.selected_key)

    def get_score(self, parameters_id, eval_params=None):
        '''Returns the score for the given parameters_id, or
        None if no score available
        Arguments:

            parameters_id (``int``):
                The parameters ID to return the score of

            eval_params (``EvaluateParametersConfig``):
                Additional parameters used for evaluation (e.g. roi,
                matching threshold, sparsity)
        '''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        score = None

        try:
            score_collection = self.database['scores']
            query = {'param_id': parameters_id}
            logger.debug("Eval params: %s", eval_params)
            if eval_params is not None:
                if isinstance(eval_params, dict):
                    query.update(eval_params)
                else:
                    query.update(eval_params.valid())
            logger.debug("Get score query: %s", query)
            old_score = score_collection.find_one(query)
            if old_score:
                del old_score['_id']
                score = old_score
                logger.debug("Loaded score for %s", query)
        finally:
            self._MongoDbGraphProvider__disconnect()
        return score

    def get_scores(self, filters=None, eval_params=None):
        '''Returns the a list of all score dictionaries or
        None if no score available
        Arguments:

            parameters_id (``int``):
                The parameters ID to return the score of

            eval_params (``EvaluateParametersConfig``):
                Additional parameters used for evaluation (e.g. roi,
                matching threshold, sparsity)
        '''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()

        try:
            if filters is not None:
                query = filters
            else:
                query = {}

            score_collection = self.database['scores']
            if eval_params is not None:
                query.update(eval_params.valid())
            logger.info("Query: %s", query)
            scores = list(score_collection.find(query))
            logger.info("Found %d scores" % len(scores))
            if len(scores) == 0:
                if "fn_div_count_unconnected_parent" in query and \
                   query["fn_div_count_unconnected_parent"] == True:
                    del query["fn_div_count_unconnected_parent"]
                if "validation_score" in query and \
                   query["validation_score"] == False:
                    del query["validation_score"]
                if "window_size" in query and \
                   query["window_size"] == 50:
                    del query["window_size"]
                if "filter_short_tracklets_len" in query and \
                   query["filter_short_tracklets_len"] == -1:
                    del query["filter_short_tracklets_len"]
                if "ignore_one_off_div_errors" in query and \
                   query["ignore_one_off_div_errors"] == False:
                    del query["ignore_one_off_div_errors"]
                logger.info("Query: %s", query)
                scores = list(score_collection.find(query))
                logger.info("Found %d scores" % len(scores))

        finally:
            self._MongoDbGraphProvider__disconnect()
        return scores

    def write_score(self, parameters_id, report, eval_params=None):
        '''Writes the score for the given parameters_id to the
        scores collection, along with the associated parameters

        Arguments:

            parameters_id (``int``):
                The parameters ID to write the score of

            report (``linajea.evaluation.Report``):
                The report with the scores to write

            eval_params (``linajea.config.EvaluateParametersConfig``):
                Additional parameters used for evaluation (e.g. roi,
                matching threshold, sparsity)
        '''
        parameters = self.get_parameters(parameters_id)
        if parameters is None:
            logger.warning("No parameters with id %s. Saving with key only",
                           str(parameters_id))

        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        try:
            score_collection = self.database['scores']
            query = {'param_id': parameters_id}
            if eval_params is not None:
                query.update(eval_params.valid())

            cnt = score_collection.count_documents(query)
            assert cnt <= 1, "multiple scores for query %s exist, don't know which to overwrite" % query

            if parameters is None:
                parameters = {}
            logger.info("writing scores for %s to %s", parameters, query)
            parameters.update(report.get_report())
            parameters.update(query)
            score_collection.replace_one(query,
                                         parameters,
                                         upsert=True)
        finally:
            self._MongoDbGraphProvider__disconnect()

    def get_nodes_roi(self):
        start = []
        end = []
        try:
            self._MongoDbGraphProvider__connect()
            self._MongoDbGraphProvider__open_db()
            self._MongoDbGraphProvider__open_collections()
            for dim in self.position_attribute:
                smallest_entry = self.nodes.find().sort([(dim, 1)]).limit(1)[0]
                start.append(smallest_entry[dim])
                largest_entry = self.nodes.find().sort([(dim, -1)]).limit(1)[0]
                end.append(largest_entry[dim] + 1)

            offset = Coordinate(start)
            end = Coordinate(end)
            size = end - offset
            nodes_roi = Roi(offset, size)
        finally:
            self._MongoDbGraphProvider__disconnect()
        return nodes_roi
