from pymongo import MongoClient, IndexModel, ASCENDING, ReplaceOne
import logging

logger = logging.getLogger(__name__)

class CandidateDatabase(object):

    def __init__(self, db_name, db_host='localhost', mode='r'):

        self.db_name = db_name
        self.db_host = db_host
        self.mode = mode
        self.client = MongoClient(host=db_host)

        existing = False
        if mode == 'w':
            # In theory, in 'w' mode, it should replace the database with a new one,
            # but I lost a lot of data this way by accident, so for now I am just warning if it exists and then
            # adding to it
            db_names = self.client.list_database_names()
            if self.db_name in db_names:
                existing = True
                logger.warning("Database with name {} already exists. New data will be added to existing data."
                               .format(self.db_name))

        self.database = self.client[db_name]
        self.nodes = self.database['nodes']
        self.edges = self.database['edges']

        if mode == 'w' and not existing:

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

    def read_nodes_and_edges(self, roi, require_selected=False, key='selected'):
        nodes = self.read_nodes(roi)
        node_ids = list([ n['id'] for n in nodes])

        edges = []

        query_size = 128 
        for i in range(0, len(node_ids), query_size):
            query = { 'source': { '$in': node_ids[i:i+query_size] }}
            if require_selected:
                query[key] = True
            edges += list(self.edges.find(query))
        
        filtered_cell_ids = set([edge['source'] for edge in edges] + 
                                [edge['target'] for edge in edges])
        filtered_cells = [cell for cell in nodes
                          if cell['id'] in filtered_cell_ids]

        return filtered_cells, edges


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

        if roi is not None:

            assert cells is not None, (
                "roi given, but no cells to check for inclusion")

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

    def update_edges(self, edges, cells=None, roi=None):

        if self.mode == 'r':
            raise RuntimeError("trying to write to read-only DB")

        if roi is not None:

            assert cells is not None, (
                "roi given, but no cells to check for inclusion")

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

            logger.debug("No edges to update.")
            return

        logger.debug("Update %d edges"%len(edges))

        self.edges.bulk_write([
            ReplaceOne(
                {
                    'source': edge['source'],
                    'target': edge['target']
                },
                edge
            )
            for edge in edges
        ])

    def read_edges(self, roi):

        nodes = self.read_nodes(roi)
        node_ids = list([ n['id'] for n in nodes])

        edges = []

        query_size = 128
        for i in range(0, len(node_ids), query_size):
            edges += list(self.edges.find({
                'source': { '$in': node_ids[i:i+query_size] }
            }))

        return edges
