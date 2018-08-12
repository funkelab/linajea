from pymongo import MongoClient, IndexModel, ASCENDING
import logging

logger = logging.getLogger(__name__)

class CandidateDatabase(object):

    def __init__(self, db_name, mode='r'):

        self.db_name = db_name
        self.mode = mode
        self.client = MongoClient()

        if mode == 'w':
            self.client.drop_database(db_name)

        self.database = self.client[db_name]
        self.nodes = self.database['nodes']
        self.edges = self.database['edges']

        if mode == 'w':

            self.nodes.create_index(
                [
                    ('t', ASCENDING),
                    ('z', ASCENDING),
                    ('y', ASCENDING),
                    ('x', ASCENDING)
                ],
                name='position')

            self.nodes.create_index(
                [
                    ('id', ASCENDING)
                ],
                name='id')

            self.edges.create_index(
                [
                    ('source', ASCENDING),
                    ('target', ASCENDING)
                ],
                name='incident')

    def write_nodes(self, nodes, roi=None):

        if self.mode == 'r':
            raise RuntimeError("trying to write to read-only DB")

        if roi is not None:

            nodes = [
                n
                for n in nodes
                if roi.contains(n['position'])
            ]

        if len(nodes) == 0:

            logger.debug("No nodes to write.")
            return

        # convert 'position' into '{t,z,y,x}'
        nodes = [
            {
                'id': int(n['id']),
                't': int(n['position'][0]),
                'z': int(n['position'][1]),
                'y': int(n['position'][2]),
                'x': int(n['position'][3]),
                'score': float(n['score'])
            }
            for n in nodes
        ]

        logger.debug("Insert %d nodes"%len(nodes))
        logger.debug(nodes)

        self.nodes.insert_many(nodes)

    def read_nodes(self, roi):
        '''Return a list of dictionaries with ``id``, ``position``, and
        ``score`` for each node in ``roi``.
        '''

        logger.debug("Querying nodes in %s", roi)

        bt, bz, by, bx = roi.get_begin()
        et, ez, ey, ex = roi.get_end()

        nodes = self.nodes.find(
            {
                't': { '$gte': bt, '$lt': et },
                'z': { '$gte': bz, '$lt': ez },
                'y': { '$gte': by, '$lt': ey },
                'x': { '$gte': bx, '$lt': ex }
            })

        # convert '{t,z,y,x}' into 'position'
        nodes = [
            {
                'id': n['id'],
                'position': (n['t'], n['z'], n['y'], n['x']),
                'score': n['score']
            }
            for n in nodes
        ]

        return nodes

    def write_edges(self, edges, cells=None, roi=None):
        '''Write edges to the DB. If ``cells`` and ``roi`` is given, restrict
        the write to edges with source nodes that have their position in
        ``roi``.

        Args:

            edges (``dict``):

                List of dicts with 'source', 'target' (forward in time), and
                'score'.

            cells (``list`` of ``dict``, optional):

                List with dicts with ``id`` and ``position``.

            roi (``daisy.Roi``, optional):

                If given, restrict writing to edges with ``source`` inside
                ``roi``.
        '''

        if self.mode == 'r':
            raise RuntimeError("trying to write to read-only DB")

        if roi is not None and cells is not None:

            cell_centers = {
                cell['id']: cell['position']
                for cell in cells
            }

            edges = [
                e
                for e in edges
                if roi.contains(cell_centers[e['source']])
            ]

        if len(edges) == 0:

            logger.debug("No edges to write.")
            return

        logger.debug("Insert %d edges"%len(edges))

        self.edges.insert_many(edges)

    def read_edges(self, roi):

        nodes = self.read_nodes(roi)
        node_ids = list([ n['id'] for n in nodes])

        edges = self.edges.find(
            {
                'source': { '$in': node_ids }
            })

        return list(edges)
