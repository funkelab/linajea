from linajea.mamut_visualization import MamutWriter, MamutMongoReader
import logging
import pymongo as mongo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_test_data(mongo_url, db_name):
    client = mongo.MongoClient(mongo_url)
    if db_name in client.list_database_names():
        logger.error("Database {} already exists. Exiting".format(db_name))
        return 1
    db = client[db_name]
    nodes = db.get_collection('nodes')
    edges = db.get_collection('edges')
    nodes.insert_one(
            {'id': 0,
             'score': 0,
             't': 0,
             'z': 1000,
             'y': 1000,
             'x': 1000}
            )
    nodes.insert_one(
            {'id': 1,
             'score': 0,
             't': 1,
             'z': 1010,
             'y': 1010,
             'x': 1010}
            )

    edges.insert_one(
            {'id': 3,
             'distance': 0,
             'target': 0,
             'source': 1,
             'score': 0,
             'selected': True}
            )


def remove_test_data(mongo_url, db_name):
    client = mongo.MongoClient(mongo_url)
    if db_name not in client.list_database_names():
        logger.error("Database {} does not exist. Cannot drop".format(db_name))
        return 1
    client.drop_database(db_name)


if __name__ == "__main__":
    mongo_url = 'mongodb://funkeAdmin:KAlSi3O8O@mongodb4.int.janelia.org'\
                    ':27023/admin?replicaSet=rsFunke'
    test_data_db_name = 'linajea_mamut_test'
    write_test_data(mongo_url, test_data_db_name)
    data = {'db_name': test_data_db_name,
            'frames': [0, 3],
            'selected_key': 'selected'
            }
    writer = MamutWriter()
    reader = MamutMongoReader(mongo_url)
    writer.add_data(reader, data)
    writer.write('140521_raw.xml', 'test_mongo_reader.xml')
    remove_test_data(mongo_url, test_data_db_name)
