import datetime
import os

import pymongo


def checkOrCreateDB(config, sample, create_if_not_found=True):
    db_host = config['general']['db_host']

    info = {}
    info["setup_dir"] = os.path.basename(config['general']['setup_dir'])
    info["iteration"] = config['prediction']['iteration']
    info["cell_score_threshold"] = config['prediction']['cell_score_threshold']
    info["sample"] = os.path.basename(sample)

    client = pymongo.MongoClient(host=db_host)
    for db_name in client.list_database_names():
        if not db_name.startswith("linajea_celegans_"):
            continue

        db = client[db_name]
        if "db_meta_info" not in db.list_collection_names():
            continue

        query_result = db["db_meta_info"].count_documents({})
        if query_result == 0:
            raise RuntimeError("invalid db_meta_info in db %s: no entry",
                               db_name)
        elif query_result > 1:
            raise RuntimeError(
                    "invalid db_meta_info in db %s: more than one entry (%d)",
                    db_name, query_result)
        else:
            assert query_result == 1
            query_result = db["db_meta_info"].find_one()
            del query_result["_id"]
            if query_result == info:
                break
    else:
        if not create_if_not_found:
            return None
        db_name = "linajea_celegans_{}".format(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                '%Y%m%d_%H%M%S'))
        client[db_name]["db_meta_info"].insert_one(info)

    return db_name
