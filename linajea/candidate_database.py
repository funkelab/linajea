from daisy.persistence import MongoDbGraphProvider
import logging
import pymongo
import time
import numpy as np
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
            node, edge, and meta collections.

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
        self.parameters_id = None
        self.selected_key = None
        if parameters_id:
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
            logger.warn("No parameters id stored or provided: cannot reset selection")
            return

        if roi:
            nodes = self.read_nodes(roi)
            node_ids = list([int(np.int64(n['id'])) for n in nodes])
            query = {self.endpoint_names[0]: {'$in': node_ids}}
        else:
            query={}

        if not parameter_ids:
            parameter_ids=[self.parameters_id]
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
        logger.info("Resetting soln for parameter_ids %s in roi %s took %d seconds",
                    parameter_ids, roi, time.time() - start_time)

    def get_parameters_id(
            self,
            tracking_parameters,
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
            params_dict = tracking_parameters.__dict__
            params_dict = {key: val
                           for key, val in params_dict.items()
                           if val is not None}
            cnt = params_collection.count_documents(params_dict)
            if fail_if_not_exists:
                assert cnt > 0, "Did not find id for parameters %s"\
                    " and fail_if_not_exists set to True" % params_dict
            if cnt > 1:
                raise RuntimeError("multiple documents found in db"
                                   " for these parameters: %s", params_dict)
            elif cnt == 1:
                find_result = params_collection.find_one(params_dict)
                logger.info("Parameters %s already in collection with id %d"
                            % (params_dict, find_result['_id']))
                params_id = find_result['_id']
            else:
                params_id = self.insert_with_next_id(params_dict,
                                                     params_collection)
                logger.info("Parameters %s not yet in collection,"
                            " adding with id %d",
                            params_dict, params_id)
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

    def set_parameters_id(self, parameters_id):
        '''Sets the parameters_id and selected_key for the CandidateDatabase,
        so that you can use reset_selection and/or get_selected_graph'''
        self.parameters_id = int(parameters_id)
        self.selected_key = 'selected_' + str(self.parameters_id)
        logger.debug("Set selected_key to %s" % self.selected_key)

    def get_score(self, parameters_id, frames=None):
        '''Returns the score for the given parameters_id, or
        None if no score available'''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        score = None

        try:
            if frames is None:
                score_collection = self.database['scores']
                old_score = score_collection.find_one({'_id': parameters_id})
            else:
                score_collection = self.database[
                    'scores'+"_".join(str(f) for f in frames)]
                old_score = score_collection.find_one(
                    {'param_id': parameters_id,
                     'frame_start': frames[0],
                     'frame_end': frames[1]})
            if old_score:
                if frames is None:
                    del old_score['_id']
                else:
                    del old_score['param_id']
                score = old_score

        finally:
            self._MongoDbGraphProvider__disconnect()
        return score

    def get_scores(self, frames=None, filters=None):
        '''Returns the a list of all score dictionaries or
        None if no score available'''
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()

        try:
            if filters is not None:
                query = filters
            else:
                query = {}

            if frames is None:
                score_collection = self.database['scores']
                scores = list(score_collection.find(query))

            else:
                score_collection = self.database[
                    'scores'+"_".join(str(f) for f in frames)]
                scores = list(score_collection.find(query))
            logger.debug("Found %d scores" % len(scores))
            if frames is None:
                for score in scores:
                    score['param_id'] = score['_id']

        finally:
            self._MongoDbGraphProvider__disconnect()
        return scores

    def write_score(self, parameters_id, report, frames=None):
        '''Writes the score for the given parameters_id to the
        scores collection, along with the associated parameters'''
        parameters = self.get_parameters(parameters_id)
        assert parameters is not None,\
            "No parameters with id %d" % parameters_id

        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        try:

            if frames is None:
                score_collection = self.database['scores']
                eval_dict = {'_id': parameters_id}
            else:
                score_collection = self.database[
                    'scores'+"_".join(str(f) for f in frames)]
                eval_dict = {'param_id': parameters_id}
            eval_dict.update(parameters)
            logger.info("%s  %s", frames, eval_dict)
            eval_dict.update(report.__dict__)
            if frames is None:
                score_collection.replace_one({'_id': parameters_id},
                                             eval_dict,
                                             upsert=True)
            else:
                eval_dict.update({'frame_start': frames[0],
                                  'frame_end': frames[1]})
                score_collection.replace_one({'param_id': parameters_id,
                                              'frame_start': frames[0],
                                              'frame_end': frames[1]},
                                             eval_dict,
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
