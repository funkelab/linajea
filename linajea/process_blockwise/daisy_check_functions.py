"""Set of daisy block-wise processing utility functions

check_function*: Check in database if block has already been processed
write_done*: Write into database that block has been processed
"""
import pymongo


def get_daisy_collection_name(step_name):
    return step_name + "_daisy"


def check_function(block, step_name, db_name, db_host):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    daisy_coll = db[get_daisy_collection_name(step_name)]
    result = daisy_coll.find_one({'_id': block.block_id[1]})
    if result is None:
        return False
    else:
        return True


def write_done(block, step_name, db_name, db_host):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    daisy_coll = db[get_daisy_collection_name(step_name)]
    daisy_coll.insert_one({'_id': block.block_id[1]})


def check_function_all_blocks(step_name, db_name, db_host):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    daisy_coll = db[get_daisy_collection_name(step_name)]
    result = daisy_coll.find_one({'_id': step_name})
    if result is None:
        return False
    else:
        return True


def write_done_all_blocks(step_name, db_name, db_host):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    daisy_coll = db[get_daisy_collection_name(step_name)]
    daisy_coll.insert_one({'_id': step_name})
