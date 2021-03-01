import datetime
import os

import pymongo


def checkOrCreateDB(db_host, setup_dir, sample, checkpoint,
                    cell_score_threshold, prefix="linajea_",
                    create_if_not_found=True):
    db_host = db_host

    info = {}
    info["setup_dir"] = os.path.basename(setup_dir)
    info["iteration"] = checkpoint
    info["cell_score_threshold"] = cell_score_threshold
    info["sample"] = os.path.basename(sample)

    return checkOrCreateDBMeta(db_host, info, prefix=prefix,
                               create_if_not_found=create_if_not_found)


def checkOrCreateDBMeta(db_host, db_meta_info, prefix="linajea_",
                        create_if_not_found=True):
    client = pymongo.MongoClient(host=db_host)
    for db_name in client.list_database_names():
        if not db_name.startswith(prefix):
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
            if query_result == db_meta_info:
                break
    else:
        if not create_if_not_found:
            return None
        db_name = "linajea_celegans_{}".format(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                '%Y%m%d_%H%M%S'))
        client[db_name]["db_meta_info"].insert_one(db_meta_info)

    return db_name
