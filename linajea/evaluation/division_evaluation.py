"""Provides a function to evaluate the recreated divisions

Can match divisions not only in the same frame as the gt divisions
but also in adjacent frames (by setting frame_buffer)
"""
import logging

import numpy as np
import scipy.spatial

from .match import match

logger = logging.getLogger(__name__)


def evaluate_divisions(
        gt_divisions,
        rec_divisions,
        target_frame,
        matching_threshold,
        frame_buffer=0,
        output_file=None):
    ''' Full frame division evaluation
    Arguments:
        gt_divisions (dict: int -> list)
            Dictionary from frame to [[z y x id], ...]

        rec_divisions (dict: int -> list)
            Dictionary from frame to [[z y x id], ...]

        target_frame (int)

        matching_threshold (int)

        frame_buffer (int):
            Number of adjacent frames that are also fully
            annotated in gt_divisions. Default is 0.

        output_file (string):
            If given, save the results to the file

    Output:
        A set of division reports with TP, FP, FN, TN, accuracy, precision
        and recall values for divisions in the target frame. Each report will
        give a different amount of leniency in the time localization of the
        division, from 0 to frame_buffer. For example, with a leniency of 1
        frame, GT divisions in the target frame can be matched to rec divisions
        in +-1 frame when calculating FN, and rec divisions in the
        target frame can be matched to GT divisions  in +-1 frame when
        calculating FP. Rec divisions in +-1 will NOT be counted as FPs however
        (because we would need GT in +-2 to confirm if they are FP or TP), and
        same for GT divisions in +-1 not counting as FNs.

        Note: if you have dense ground truth, it is better to do a global
        hungarian matching with whatever leniency you want, and then calculate
        a global report, not one focusing on a single frame.

    Algorithm:
        Make a KD tree for each division point in each frame of GT and REC.
        Match GT and REC divisions in target_frame using hungarian matching,
            calculate report.
        For unmatched GT divisions in target_frame, try matching to rec in +-1.
        For unmatched REC divisions in target_frame, try matching to GT in +-1.
        Calculate updated report.
        Repeat last 2 lines until leniency reaches frame buffer.
    '''
    gt_kd_trees = {}
    gt_node_ids = {}
    rec_kd_trees = {}
    rec_node_ids = {}

    for b in range(0, frame_buffer + 1):
        if b == 0:
            ts = [target_frame]
        else:
            ts = [target_frame - b, target_frame + b]
        rec_nodes = []
        gt_nodes = []
        rec_positions = []
        gt_positions = []
        for t in ts:
            if t in rec_divisions:
                # ids only
                rec_nodes.extend([n[3] for n in rec_divisions[t]])
                # positions only
                rec_positions.extend([n[0:3] for n in rec_divisions[t]])

            if t in gt_divisions:
                # ids only
                gt_nodes.extend([n[3] for n in gt_divisions[t]])
                # positions only
                gt_positions.extend([n[0:3] for n in gt_divisions[t]])

        if len(gt_positions) > 0:
            gt_kd_trees[b] = scipy.spatial.cKDTree(gt_positions)
        else:
            gt_kd_trees[b] = None
        gt_node_ids[b] = gt_nodes
        if len(rec_positions) > 0:
            rec_kd_trees[b] = scipy.spatial.cKDTree(rec_positions)
        else:
            rec_kd_trees[b] = None
        rec_node_ids[b] = rec_nodes

    matches = []
    gt_target_tree = gt_kd_trees[0]
    rec_target_tree = rec_kd_trees[0]
    logger.debug("Node Ids gt: %s", gt_node_ids)
    logger.debug("Node Ids rec: %s", rec_node_ids)
    reports = []
    for b in range(0, frame_buffer + 1):
        if b == 0:
            # Match GT and REC divisions in target_frame using hungarian
            # matching, calculate report.
            costs = construct_costs(gt_target_tree, gt_node_ids[0],
                                    rec_target_tree, rec_node_ids[0],
                                    matching_threshold)
            if len(costs) == 0:
                matches = []
            else:
                matches, soln_cost = match(costs,
                                           matching_threshold + 1)
            logger.info("found %d matches in target frame" % len(matches))
            report = calculate_report(gt_node_ids, rec_node_ids, matches)
            reports.append(report)
            logger.info("report in target frame: %s", report)
        else:
            # For unmatched GT divisions in target_frame,
            # try matching to rec in +-b.
            matched_gt = [m[0] for m in matches]
            gt_costs = construct_costs(
                    gt_target_tree, gt_node_ids[0],
                    rec_kd_trees[b], rec_node_ids[b],
                    matching_threshold,
                    exclude_gt=matched_gt)
            if len(gt_costs) == 0:
                gt_matches = []
            else:
                gt_matches, soln_cost = match(
                        gt_costs,
                        matching_threshold + 1)
            logger.info("Found %d gt matches in frames +-%d",
                        len(gt_matches), b)
            matches.extend(gt_matches)
            # For unmatched REC divisions in target_frame,
            # try matching to GT in +-b.
            matched_rec = [m[1] for m in matches]
            rec_costs = construct_costs(
                    gt_kd_trees[b], gt_node_ids[b],
                    rec_target_tree, rec_node_ids[0],
                    matching_threshold,
                    exclude_rec=matched_rec)
            if len(rec_costs) == 0:
                rec_matches = []
            else:
                rec_matches, soln_cost = match(
                        rec_costs,
                        matching_threshold + 1)
            logger.info("Found %d rec matches in frames +-%d",
                        len(rec_matches), b)
            matches.extend(rec_matches)

            # Calculate updated report.
            report = calculate_report(gt_node_ids, rec_node_ids, matches)
            reports.append(report)
            logger.info("report +-%d: %s", b, report)
    if output_file:
        save_results_to_file(reports, output_file)
    return reports


def construct_costs(
        gt_tree, gt_nodes,
        rec_tree, rec_nodes,
        matching_threshold,
        exclude_gt=[],
        exclude_rec=[]):
    costs = {}
    if gt_tree is None or rec_tree is None:
        return costs
    neighbors = gt_tree.query_ball_tree(rec_tree, matching_threshold)
    for i, js in enumerate(neighbors):
        gt_node = gt_nodes[i]
        if gt_node in exclude_gt:
            continue
        for j in js:
            rec_node = rec_nodes[j]
            if rec_node in exclude_rec:
                continue
            distance = np.linalg.norm(
                np.array(gt_tree.data[i]) -
                np.array(rec_tree.data[j]))
            costs[(gt_node, rec_node)] = distance
    return costs


def calculate_report(
        gt_node_ids,
        rec_node_ids,
        matches):
    matched_gt = [m[0] for m in matches]
    matched_rec = [m[1] for m in matches]

    # gt_total, rec_total, FP, FN, Prec, Rec, F1
    gt_target_divs = gt_node_ids[0]
    rec_target_divs = rec_node_ids[0]
    gt_total = len(gt_target_divs)
    rec_total = len(rec_target_divs)
    fp_nodes = [n for n in rec_target_divs
                if n not in matched_rec]
    fp = len(fp_nodes)
    fn_nodes = [n for n in gt_target_divs
                if n not in matched_gt]
    fn = len(fn_nodes)
    prec = (rec_total - fp) / rec_total if rec_total > 0 else None
    rec = (gt_total - fn) / gt_total if gt_total > 0 else None
    f1 = (2 * prec * rec / (prec + rec)
          if prec is not None and rec is not None and prec + rec > 0
          else None)
    return (gt_total, rec_total, fp, fn, prec, rec, f1)


def save_results_to_file(reports, filename):
    header = "frames, gt_total, rec_total, FP, FN, Prec, Rec, F1\n"
    with open(filename, 'w') as f:
        f.write(header)
        for frames, report in enumerate(reports):
            f.write(str(frames))
            f.write(", ")
            f.write(", ".join(list(map(str, report))))
            f.write("\n")
