from daisy.persistence import MongoDbGraphProvider
import logging
import pymongo

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

    def reset_selection(self):
        ''' Removes all selections for self.parameters_id from mongodb
        edges collection
        '''
        if self.parameters_id is None:
            logger.warn("No parameters id: cannot reset selection")
            return

        logger.info("Resetting solution for parameters_id %s"
                    % self.parameters_id)
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        edge_coll = self.database['edges']
        edge_coll.update_many({}, {'$unset': {self.selected_key: ""}})
        daisy_coll_name = 'solve_' + str(self.parameters_id) + '_daisy'
        self.database.drop_collection(daisy_coll_name)
        logger.info("Done resetting solution for parameters_id %s"
                    % self.parameters_id)

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
            find_result = params_collection.find_one(params_dict)
            if fail_if_not_exists:
                assert find_result, "Did not find id for parameters %s"\
                    " and fail_if_not_exists set to True" % params_dict
            if find_result:
                logger.info("Parameters %s already in collection with id %d"
                            % (params_dict, find_result['_id']))
                params_id = find_result['_id']
            else:
                params_id = self.insert_with_next_id(params_dict,
                                                     params_collection)
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
            score_collection = self.database['scores']
            if frames is None:
                old_score = score_collection.find_one({'_id': parameters_id})
            else:
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

    def write_score(self, parameters_id, report, frames=None):
        '''Writes the score for the given parameters_id to the
        scores collection, along with the associated parameters'''
        parameters = self.get_parameters(parameters_id)
        assert parameters is not None,\
            "No parameters with id %d" % parameters_id

        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        try:

            score_collection = self.database['scores']
            if frames is None:
                eval_dict = {'_id': parameters_id}
            else:
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
                res = score_collection.replace_one({'param_id': parameters_id,
                                              'frame_start': frames[0],
                                              'frame_end': frames[1]},
                                             eval_dict,
                                             upsert=True)
        finally:
            self._MongoDbGraphProvider__disconnect()
