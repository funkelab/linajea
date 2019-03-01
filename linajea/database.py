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
            endpoint_names=['source', 'target']):
        super().__init__(
                db_name,
                host=mongo_url,
                mode=mode,
                total_roi=total_roi,
                directed=True,
                position_attribute=['t', 'z', 'y', 'x'],
                endpoint_names=endpoint_names
                )

    def get_nodes_and_edges(
            self,
            roi,
            require_selected=False,
            key='selected'):
        nodes = self.read_nodes(roi)
        edges = self.read_edges(roi, nodes=nodes)
        if require_selected:
            filtered_edges = []
            for edge in edges:
                if key in edge and edge[key]:
                    filtered_edges.append(edge)
            u, v = self.endpoint_names
            filtered_cell_ids = set([edge[u] for edge in filtered_edges] +
                                    [edge[v] for edge in filtered_edges])
            filtered_cells = [cell for cell in nodes
                              if cell['id'] in filtered_cell_ids]
            return filtered_cells, filtered_edges
        else:
            return nodes, edges

    def get_parameters_id(self, tracking_parameters, fail_if_not_exists=False):
        '''Get id for parameter set from mongo collection.
        If fail_if_not_exists, fail if the parameter set isn't already there.
        The default is to assign a new id and write it to the collection.'''
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

        return params_id

    def insert_with_next_id(self, document, collection):
        count_coll = self.database['meta']

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

    def get_result(self, parameters_id):
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()

        result = None
        try:
            results_collection = self.database['results']

            track_doc = results_collection.find_one(
                    {'_id': 'tracks_' + parameters_id})
            cell_doc = results_collection.find_one(
                    {'_id': 'cells_' + parameters_id})
            stats_doc = results_collection.find_one(
                    {'_id': 'stats_' + parameters_id})
            if track_doc and cell_doc and stats_doc:
                logger.info((
                        "Matching tracks already saved for {}. "
                        "Reading from matches from db.")
                        .format(parameters_id))
                del track_doc['_id']
                track_matches = track_doc.values()

                del cell_doc['_id']
                cell_matches = cell_doc.values()

                s = stats_doc['splits']
                m = stats_doc['merges']
                fp = stats_doc['fp']
                fn = stats_doc['fn']
                result = (track_matches, cell_matches, s, m, fp, fn)
        finally:
            self._MongoDbGraphProvider__disconnect()

        return result

    def write_result(
            self,
            parameters_id,
            track_matches,
            cell_matches,
            splits,
            merges,
            false_positives,
            false_negatives):

        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()

        try:
            results_collection = self.database['results']

            results_collection.insert_one({
                'splits': splits,
                'merges': merges,
                'fp': false_positives,
                'fn': false_negatives,
                '_id': 'stats_' + parameters_id})
            track_matches_doc = {}
            for i in range(len(track_matches)):
                track_matches_doc[str(i)] = track_matches[i]
            track_matches_doc['_id'] = 'tracks_' + parameters_id
            results_collection.insert_one(track_matches_doc)

            cell_matches_doc = {}
            for i in range(len(cell_matches)):
                cell_matches_doc[str(i)] = cell_matches[i]
            cell_matches_doc['_id'] = 'cells_' + parameters_id
            results_collection.insert_one(cell_matches_doc)
        finally:
            self._MongoDbGraphProvider__disconnect()

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

    def write_score(self, parameters_id, parameters, score):
        self._MongoDbGraphProvider__connect()
        self._MongoDbGraphProvider__open_db()

        try:

            score_collection = self.database['scores']
            eval_dict = {'_id': parameters_id}
            eval_dict.update(parameters.__dict__)
            eval_dict.update(score.__dict__)
            score_collection.insert_one(eval_dict)
        finally:
            self._MongoDbGraphProvider__disconnect()
