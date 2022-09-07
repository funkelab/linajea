import unittest
import random

import networkx as nx

from linajea.evaluation.validation_metric import _split_into_tracks


class TestSplitIntoTracks(unittest.TestCase):

    def test_remove_unconnected_node(self):
        graph = self.create_division()
        conn_components = _split_into_tracks(graph)
        self.assertEqual(len(conn_components), 2)

    def create_division(self):
        zyx_range = [0, 10]
        track = nx.DiGraph()
        track.add_node(
                0,
                t=0,
                z=random.randint(*zyx_range),
                y=random.randint(*zyx_range),
                x=random.randint(*zyx_range))
        track.add_node(
                1,
                t=1,
                z=random.randint(*zyx_range),
                y=random.randint(*zyx_range),
                x=random.randint(*zyx_range))
        track.add_node(
                2,
                t=1,
                z=random.randint(*zyx_range),
                y=random.randint(*zyx_range),
                x=random.randint(*zyx_range))
        track.add_edge(1, 0)
        track.add_edge(2, 0)
        return track
