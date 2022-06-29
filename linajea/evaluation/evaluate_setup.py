import logging
import os
import sys
import time

import networkx as nx
import numpy as np
import scipy.spatial

import daisy
import funlib.math

import linajea.tracking
import linajea.utils
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def evaluate_setup(linajea_config):

    assert len(linajea_config.solve.parameters) == 1, \
        "can only handle single parameter set"
    parameters = linajea_config.solve.parameters[0]

    data = linajea_config.inference_data.data_source
    db_name = data.db_name
    db_host = linajea_config.general.db_host
    logger.debug("ROI used for evaluation: %s %s",
                 linajea_config.evaluate.parameters.roi, data.roi)
    if linajea_config.evaluate.parameters.roi is not None:
        assert linajea_config.evaluate.parameters.roi.shape[0] <= data.roi.shape[0], \
            "your evaluation ROI is larger than your data roi!"
        data.roi = linajea_config.evaluate.parameters.roi
    else:
        linajea_config.evaluate.parameters.roi = data.roi
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
            score = {}
            for k, v in old_score.items():
                if not isinstance(k, list) or k != "roi":
                    score[k] = v
            logger.info("Stored results: %s", score)
            return

    logger.info("Evaluating %s in %s",
                os.path.basename(data.datafile.filename)
                if data.datafile is not None else db_name, evaluate_roi)

    edges_db = linajea.utils.CandidateDatabase(db_name, db_host,
                                               parameters_id=parameters_id)

    logger.info("Reading cells and edges in db %s with parameter_id %d"
                % (db_name, parameters_id))
    start_time = time.time()
    subgraph = edges_db.get_selected_graph(evaluate_roi)

    logger.info("Read %d cells and %d edges in %s seconds"
                % (subgraph.number_of_nodes(),
                   subgraph.number_of_edges(),
                   time.time() - start_time))

    if linajea_config.general.two_frame_edges:
        subgraph = split_two_frame_edges(linajea_config, subgraph, evaluate_roi)

    if subgraph.number_of_edges() == 0:
        logger.warn("No selected edges for parameters_id %d. Skipping"
                    % parameters_id)
        return

    if linajea_config.evaluate.parameters.filter_polar_bodies or \
       linajea_config.evaluate.parameters.filter_polar_bodies_key:
        subgraph = filter_polar_bodies(linajea_config, subgraph,
                                       edges_db, evaluate_roi)

    subgraph = maybe_filter_short_tracklets(linajea_config, subgraph,
                                            evaluate_roi)

    track_graph = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)

    gt_db = linajea.utils.CandidateDatabase(
        linajea_config.inference_data.data_source.gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s"
                % linajea_config.inference_data.data_source.gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db.get_graph(
        evaluate_roi,
        subsampling=linajea_config.general.subsampling,
        subsampling_seed=linajea_config.general.subsampling_seed)
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))

    if linajea_config.inference_data.data_source.gt_db_name_polar is not None and \
       not linajea_config.evaluate.parameters.filter_polar_bodies and \
       not linajea_config.evaluate.parameters.filter_polar_bodies_key:
        gt_subgraph = add_gt_polar_bodies(linajea_config, gt_subgraph,
                                          db_host, evaluate_roi)

    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %d" % parameters_id)
    matching_threshold = linajea_config.evaluate.parameters.matching_threshold
    validation_score = linajea_config.evaluate.parameters.validation_score
    ignore_one_off_div_errors = \
        linajea_config.evaluate.parameters.ignore_one_off_div_errors
    fn_div_count_unconnected_parent = \
        linajea_config.evaluate.parameters.fn_div_count_unconnected_parent
    window_size=linajea_config.evaluate.parameters.window_size
    report = evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=matching_threshold,
            sparse=linajea_config.general.sparse,
            validation_score=validation_score,
            window_size=window_size,
            ignore_one_off_div_errors=ignore_one_off_div_errors,
            fn_div_count_unconnected_parent=fn_div_count_unconnected_parent)

    logger.info("Done evaluating results for %d. Saving results to mongo."
                % parameters_id)
    logger.info("Result summary: %s", report.get_short_report())
    results_db.write_score(parameters_id, report,
                           eval_params=linajea_config.evaluate.parameters)
    res = report.get_short_report()
    print("|              |  fp |  fn |  id | fp_div | fn_div | sum_div |"
          " sum | DET | TRA |   REFT |     NR |     ER |    GT |")
    sum_errors = (res['fp_edges'] + res['fn_edges'] +
                  res['identity_switches'] +
                  res['fp_divisions'] + res['fn_divisions'])
    sum_divs = res['fp_divisions'] + res['fn_divisions']
    reft = res["num_error_free_tracks"]/res["num_gt_cells_last_frame"]
    print("| {:3d} | {:3d} | {:3d} |"
          "    {:3d} |    {:3d} |     {:3d} | {:3d} |     |     |"
          " {:.4f} | {:.4f} | {:.4f} | {:5d} |".format(
              int(res['fp_edges']), int(res['fn_edges']),
              int(res['identity_switches']),
              int(res['fp_divisions']), int(res['fn_divisions']),
              int(sum_divs),
              int(sum_errors),
              reft, res['node_recall'], res['edge_recall'],
              int(res['gt_edges'])))


def split_two_frame_edges(linajea_config, subgraph, evaluate_roi):
    voxel_size = daisy.Coordinate(linajea_config.inference_data.data_source.voxel_size)
    cells_by_frame = {}
    for cell in subgraph.nodes:
        cell = subgraph.nodes[cell]
        t = cell['t']
        if t not in cells_by_frame:
            cells_by_frame[t] = []
        cell_pos = np.array([cell['z'], cell['y'], cell['x']])
        cells_by_frame[t].append(cell_pos)
    kd_trees_by_frame = {}
    for t, cells in cells_by_frame.items():
        kd_trees_by_frame[t] = scipy.spatial.cKDTree(cells)
    edges = list(subgraph.edges)
    for u_id, v_id in edges:
        u = subgraph.nodes[u_id]
        v = subgraph.nodes[v_id]
        frame_diff = abs(u['t']-v['t'])
        if frame_diff == 1:
            continue
        elif frame_diff == 0:
            raise RuntimeError("invalid edges? no diff in t %s %s", u, v)
        elif frame_diff == 2:
            u_pos = np.array([u['z'], u['y'], u['x']])
            v_pos = np.array([v['z'], v['y'], v['x']])
            w = {}
            for k in u.keys():
                if k not in v.keys():
                    continue
                if "probable_gt" in k:
                    continue
                u_v = u[k]
                v_v = v[k]

                if "parent_vector" in k:
                    u_v = np.array(u_v)
                    v_v = np.array(v_v)
                w[k] = (u_v + v_v)/2
                if "parent_vector" in k:
                    w[k] = list(w[k])
            w['t'] = u['t'] - 1
            w_id = int(funlib.math.cantor_number(
                [i1/i2+i3
                 for i1,i2,i3 in zip(evaluate_roi.get_begin(),
                                     voxel_size,
                                     [w['t'], w['z'], w['y'], w['x']])]))
            d, i = kd_trees_by_frame[w['t']].query(
                np.array([w['z'], w['y'], w['x']]))
            print("inserted cell from two-frame-edge {} {} -> {}".format(
                (u_id, u['t'], u_pos), (v_id, v['t'], v_pos), (w_id, [w['t'], w['z'], w['y'], w['x']])))
            subgraph.remove_edge(u_id, v_id)
            if d <= 7:
                print("inserted cell already exists {}".format(cells_by_frame[w['t']][i]))
            else:
                subgraph.add_node(w_id, **w)
                subgraph.add_edge(u_id, w_id)
                subgraph.add_edge(w_id, v_id)
        else:
            raise RuntimeError("invalid edges? diff of %s in t %s %s", frame_diff, u, v)

    logger.info("After splitting two_frame edges: %d cells and %d edges"
                % (subgraph.number_of_nodes(),
                   subgraph.number_of_edges()))

    return subgraph


def filter_polar_bodies(linajea_config, subgraph, edges_db, evaluate_roi):
    logger.debug("%s %s",
                 linajea_config.evaluate.parameters.filter_polar_bodies,
                 linajea_config.evaluate.parameters.filter_polar_bodies_key)
    if not linajea_config.evaluate.parameters.filter_polar_bodies and \
       linajea_config.evaluate.parameters.filter_polar_bodies_key is not None:
        pb_key = linajea_config.evaluate.parameters.filter_polar_bodies_key
    else:
        pb_key = linajea_config.solve.parameters[0].cell_cycle_key + "polar"

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
        # else:
        #     for ch in cnt_polar_uninterrupted:
        #         if len(ch) > 5:
        #             subgraph.remove_nodes_from(ch)
        #             logger.info("removing %s potential polar nodes (2)", len(ch))

    return subgraph

def maybe_filter_short_tracklets(linajea_config, subgraph, evaluate_roi):
    track_graph_tmp = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)
    last_frame = evaluate_roi.get_end()[0]-1
    for track in track_graph_tmp.get_tracks():
        min_t = 9999
        max_t = -1
        for node_id, node in track.nodes(data=True):
            if node['t'] < min_t:
                min_t = node['t']
            if node['t'] > max_t:
                max_t = node['t']

        logger.info("track begin: {}, track end: {}, track len: {}".format(
            min_t, max_t, len(track.nodes())))

        if len(track.nodes()) < linajea_config.evaluate.parameters.filter_short_tracklets_len \
           and max_t != last_frame:
            logger.info("removing %s nodes (very short tracks < %d)",
                        len(track.nodes()),
                        linajea_config.evaluate.filter_short_tracklets_len)
            subgraph.remove_nodes_from(track.nodes())

    return subgraph


def add_gt_polar_bodies(linajea_config, gt_subgraph, db_host, evaluate_roi):
    logger.info("polar bodies are not filtered, adding polar body GT..")
    gt_db_polar = linajea.utils.CandidateDatabase(
        linajea_config.inference_data.data_source.gt_db_name_polar, db_host)
    gt_polar_subgraph = gt_db_polar[evaluate_roi]
    gt_mx_id = max(gt_subgraph.nodes()) + 1
    mapping = {n: n+gt_mx_id for n in gt_polar_subgraph.nodes()}
    gt_polar_subgraph = nx.relabel_nodes(gt_polar_subgraph, mapping,
                                         copy=False)
    gt_subgraph.update(gt_polar_subgraph)

    logger.info("Read %d cells and %d edges in %s seconds (after adding polar bodies)"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))

    return gt_subgraph
