import unittest
import random
import networkx as nx
from linajea.evaluation.validation_metric import (
        track_distance, norm_distance, validation_score)
import logging

logging.basicConfig(level=logging.INFO)


class TestValidationMetric(unittest.TestCase):
    tolerance_places = 4

    # One GT, one rec track
    def test_perfect(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        self.assertAlmostEqual(
                track_distance(gt_track, rec_track), 0,
                places=self.tolerance_places)

    def test_empty_reconstruction(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = nx.DiGraph()
        self.assertEqual(track_distance(gt_track, rec_track), length)

    def test_one_off(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        for node_id in range(length):
            rec_track.nodes[node_id]['x'] += 1
        self.assertAlmostEqual(
                track_distance(gt_track, rec_track),
                length * norm_distance(1),
                places=self.tolerance_places)

    def test_far_away(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        for node_id in range(length):
            rec_track.nodes[node_id]['x'] += 200
        self.assertAlmostEqual(
                track_distance(gt_track, rec_track),
                length,
                places=self.tolerance_places)

    def test_missing_first_edge(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        rec_track.remove_node(0)
        self.assertAlmostEqual(
                track_distance(gt_track, rec_track),
                1,
                places=self.tolerance_places)

    def test_extra_edge(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length + 1, seed=seed)
        self.assertAlmostEqual(
                track_distance(gt_track, rec_track),
                1,
                places=self.tolerance_places)

    # One GT, Multiple Rec
    def test_missing_middle_edge(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        rec_track.remove_edge(4, 3)
        self.assertAlmostEqual(
                validation_score(gt_track, rec_track),
                min([len(track) for track in
                     nx.weakly_connected_components(rec_track)]),
                places=self.tolerance_places)

    def test_fp_div(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        rec_track.add_node(length, t=4, x=0, y=0, z=0)
        rec_track.add_edge(length, 3)
        self.assertAlmostEqual(
                validation_score(gt_track, rec_track), 3,
                places=self.tolerance_places)

    def test_choosing_closer(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        for node_id in range(length):
            rec_track.nodes[node_id]['x'] += 1
        second_rec_track = self.create_track(length, seed=seed, min_id=length)
        for node_id in range(length, length*2):
            second_rec_track.nodes[node_id]['x'] += 5
        self.assertAlmostEqual(
                validation_score(gt_track, rec_track),
                length * norm_distance(1),
                places=self.tolerance_places)

    def test_choosing_continuous(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        rec_track.remove_edge(4, 3)
        second_rec_track = self.create_track(length, seed=seed, min_id=length)
        for node_id in range(length, 2*length):
            second_rec_track.nodes[node_id]['x'] += 10
        rec_track = nx.union(rec_track, second_rec_track)
        shorter_segment_len = min([len(track) for track in
                                   nx.weakly_connected_components(rec_track)])
        self.assertAlmostEqual(
                validation_score(gt_track, rec_track),
                min(shorter_segment_len,
                    length * norm_distance(10)),
                places=self.tolerance_places)

    # Multiple GT, One Rec

    def test_missed_division(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        loc = rec_track.nodes[4]
        gt_track.add_node(length, t=4,
                          x=loc['x'], y=loc['y'], z=loc['z'])
        gt_track.add_edge(length, 3)
        self.assertAlmostEqual(
                validation_score(gt_track, rec_track),
                length*2 - 3,
                places=self.tolerance_places)

    def test_reusing_rec(self):
        length = 8
        seed = 1
        gt_track = self.create_track(length, seed=seed)
        rec_track = self.create_track(length, seed=seed)
        second_gt_track = self.create_track(length, seed=seed, min_id=length)
        for node_id in range(length, length*2):
            second_gt_track.nodes[node_id]['x'] += 10
        gt_track = nx.union(gt_track, second_gt_track)
        self.assertAlmostEqual(
                validation_score(gt_track, rec_track),
                length * norm_distance(10),
                places=self.tolerance_places)

    # track creation helper
    def create_track(self, length, seed=None, min_id=0):
        zyx_range = [0, 10]
        t_range = [0, length]
        track = nx.DiGraph()
        if seed:
            random.seed(seed)
        for t in range(*t_range):
            node_id = t + min_id
            track.add_node(
                    node_id,
                    t=t,
                    z=random.randint(*zyx_range),
                    y=random.randint(*zyx_range),
                    x=random.randint(*zyx_range))
            if t > t_range[0]:
                track.add_edge(node_id, node_id-1)
        return track
