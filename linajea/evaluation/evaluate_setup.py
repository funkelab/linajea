import logging
import os
import sys
import time

import networkx as nx

import daisy

import linajea.tracking
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def evaluate_setup(linajea_config):

    assert len(linajea_config.solve.parameters) == 1, \
        "can only handle single parameter set"
    parameters = linajea_config.solve.parameters[0]

    data = linajea_config.inference.data_source
    db_name = data.db_name
    db_host = linajea_config.general.db_host
    evaluate_roi = daisy.Roi(offset=data.roi.offset,
                             shape=data.roi.shape)

    # determine parameters id from database
    results_db = linajea.CandidateDatabase(db_name, db_host)
    parameters_id = results_db.get_parameters_id(parameters)

    if not linajea_config.evaluate.from_scratch:
        old_score = results_db.get_score(parameters_id,
                                         linajea_config.evaluate.parameters)
        if old_score:
            logger.info("Already evaluated %d (%s). Skipping" %
                        (parameters_id, linajea_config.evaluate.parameters))
            logger.info("Stored results: %s", old_score)
            return

    logger.info("Evaluating %s in %s",
                os.path.basename(data.datafile.filename), evaluate_roi)

    edges_db = linajea.CandidateDatabase(db_name, db_host,
                                         parameters_id=parameters_id)

    logger.info("Reading cells and edges in db %s with parameter_id %d"
                % (db_name, parameters_id))
    start_time = time.time()
    subgraph = edges_db.get_selected_graph(evaluate_roi)

    logger.info("Read %d cells and %d edges in %s seconds"
                % (subgraph.number_of_nodes(),
                   subgraph.number_of_edges(),
                   time.time() - start_time))

    if subgraph.number_of_edges() == 0:
        logger.warn("No selected edges for parameters_id %d. Skipping"
                    % parameters_id)
        return

    if linajea_config.evaluate.parameters.filter_polar_bodies or \
       linajea_config.evaluate.parameters.filter_polar_bodies_key:
        logger.debug("%s %s",
                     linajea_config.evaluate.parameters.filter_polar_bodies,
                     linajea_config.evaluate.parameters.filter_polar_bodies_key)
        if not linajea_config.evaluate.parameters.filter_polar_bodies and \
           linajea_config.evaluate.parameters.filter_polar_bodies_key is not None:
            pb_key = linajea_config.evaluate.parameters.filter_polar_bodies_key
        else:
            pb_key = parameters.cell_cycle_key + "polar"

        # temp. remove edges from mother to daughter cells to split into chains
        tmp_subgraph = edges_db.get_selected_graph(evaluate_roi)
        for node in list(tmp_subgraph.nodes()):
            if tmp_subgraph.degree(node) > 2:
                es = list(tmp_subgraph.predecessors(node))
                tmp_subgraph.remove_edge(es[0], node)
                tmp_subgraph.remove_edge(es[1], node)
        rec_graph = linajea.tracking.TrackGraph(
            tmp_subgraph, frame_key='t', roi=tmp_subgraph.roi)

        # for each chain
        for track in rec_graph.get_tracks():
            cnt_nodes = 0
            cnt_polar = 0
            cnt_polar_uninterrupted = [[]]
            nodes = []
            for node_id, node in track.nodes(data=True):
                nodes.append((node['t'], node_id, node))

            # check if > 50% are polar bodies
            nodes = sorted(nodes)
            for _, node_id, node in nodes:
                cnt_nodes += 1
                try:
                    if node[pb_key] > 0.5:
                        cnt_polar += 1
                        cnt_polar_uninterrupted[-1].append(node_id)
                    else:
                        cnt_polar_uninterrupted.append([])
                except KeyError:
                    pass

            # then remove
            if cnt_polar/cnt_nodes > 0.5:
                subgraph.remove_nodes_from(track.nodes())
                logger.info("removing %s potential polar nodes",
                            len(track.nodes()))

    track_graph = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)

    gt_db = linajea.CandidateDatabase(
        linajea_config.inference.data_source.gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s"
                % linajea_config.inference.data_source.gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db[evaluate_roi]
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))

    if linajea_config.inference.data_source.gt_db_name_polar is not None and \
       not linajea_config.evaluate.parameters.filter_polar_bodies and \
       not linajea_config.evaluate.parameters.filter_polar_bodies_key:
        logger.info("polar bodies are not filtered, adding polar body GT..")
        gt_db_polar = linajea.CandidateDatabase(
            linajea_config.inference.data_source.gt_db_name_polar, db_host)
        gt_polar_subgraph = gt_db_polar[evaluate_roi]
        gt_mx_id = max(gt_subgraph.nodes()) + 1
        mapping = {n: n+gt_mx_id for n in gt_polar_subgraph.nodes()}
        gt_polar_subgraph = nx.relabel_nodes(gt_polar_subgraph, mapping,
                                             copy=False)
        gt_subgraph.update(gt_polar_subgraph)

    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %d" % parameters_id)
    report = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=linajea_config.evaluate.parameters.matching_threshold,
            sparse=linajea_config.general.sparse,
            validation_score=linajea_config.evaluate.parameters.validation_score,
            window_size=linajea_config.evaluate.parameters.window_size)

    logger.info("Done evaluating results for %d. Saving results to mongo."
                % parameters_id)
    logger.info("Result summary: %s", report.get_short_report())
    results_db.write_score(parameters_id, report,
                           eval_params=linajea_config.evaluate.parameters)
