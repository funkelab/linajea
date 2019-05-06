from daisy.persistence import MongoDbGraphProvider
import logging
import pymongo

logger = logging.getLogger(__name__)


class CandidateDatabase(MongoDbGraphProvider):
    '''Wrapper for daisy mongo graph that allows storing
    ILP paramter sets, matched tracks, and scores.'''

    def __init__(
            self,
            db_name,
            mongo_url,
            mode='r',
            total_roi=None,
            endpoint_names=['source', 'target'],
            parameters_id=None):

        super().__init__(
                db_name,
                host=mongo_url,
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

    def get_selected_graph(
            self,
            roi):
        subgraph = self.get_graph(
                roi,
                edges_filter={self.selected_key: True},
                edge_attrs=[self.selected_key])
        unattached_nodes = [node for node in subgraph.nodes()
                            if subgraph.degree(node) == 0]
        subgraph.remove_nodes_from(unattached_nodes)
        return subgraph

    def reset_selection(self):
        if self.parameters_id is None:
            logger.warn("No parameters id: cannot reset selection")
            return

        logger.info("Resetting solution for parameters_id %s"
                    % self.parameters_id)
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        edge_coll = self.database['edges']
        edge_coll.update_many({}, {'$unset': {self.selected_key: ""}})
        logger.info("Done resetting solution for parameters_id %s"
                    % self.parameters_id)

    # TODO: Should block size and context be part of tracking params?
    def get_parameters_id(
            self,
            tracking_parameters,
            block_size,
            context,
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
            params_dict['block_size'] = block_size
            params_dict['context'] = context
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
        '''Returns null if there are no parameters with the given id'''
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
        self.parameters_id = int(parameters_id)
        self.selected_key = 'selected_' + str(self.parameters_id)

    def get_score(self, parameters_id):
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        score = None

        try:
            score_collection = self.database['scores']
            old_score = score_collection.find_one({'_id': parameters_id})
            if old_score:
                del old_score['_id']
                score = old_score

        finally:
            self._MongoDbGraphProvider__disconnect()
        return score

    def write_score(self, parameters_id, score):
        parameters = self.get_parameters(parameters_id)
        assert parameters is not None,\
            "No parameters with id %d" % parameters_id

        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()
        try:

            score_collection = self.database['scores']
            eval_dict = {'_id': parameters_id}
            eval_dict.update(parameters)
            eval_dict.update(score.__dict__)
            score_collection.replace_one({'_id': parameters_id},
                                         eval_dict,
                                         upsert=True)
        finally:
            self._MongoDbGraphProvider__disconnect()
