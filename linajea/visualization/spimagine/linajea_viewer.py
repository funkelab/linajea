from PyQt5 import QtWidgets
import colorsys
import daisy
import logging
import math
import networkx as nx
import numpy as np
import spimagine
import sys

logger = logging.getLogger(__name__)


class LinajeaViewer:

    def __init__(
            self,
            raw,
            rec_graph_provider,
            gt_graph_provider,
            roi,
            selected_key='selected',
            channel=0):

        self.raw = raw
        self.rec_graph = (
            rec_graph_provider[roi]
            if rec_graph_provider else None)
        self.gt_graph = (
            gt_graph_provider[roi]
            if gt_graph_provider else None)
        self.roi = roi
        self.selected_key = selected_key
        self.channel = 0

        if self.rec_graph:
            self.__propagate_selected_attributes(self.rec_graph)
            self.__compute_spimagine_pos(self.rec_graph)
            self.__label_tracks(self.rec_graph)
        if self.gt_graph:
            self.__compute_spimagine_pos(self.gt_graph)

        self.default_color = (0.2, 0.2, 0.2)
        self.selected_color = (0.1, 1.0, 0.1)
        self.nonselected_color = (1.0, 0.2, 0.5)

        self.show_edge_attributes = []
        self.show_node_ids = False
        self.show_node_scores = True
        self.hide_non_selected = False
        self.show_track_colors = False

        self.limit_to_frames = (0, 0)
        self.prev_t = -1
        self.t = roi.get_begin()[0]

        self.viewer = None

    def show(self):

        self.app = QtWidgets.QApplication(sys.argv)
        self.viewer = self.__create_viewer()
        self.draw_annotations()

        self.__block_until_closed()

    def clear_annotations(self):

        self.viewer.glWidget.meshes = []
        self.viewer.glWidget.lines = []
        self.viewer.glWidget.texts = []

    def draw_annotations(self):

        self.limit_to_frames = (self.t - 0, self.t + 2)

        logger.debug("showing annotations in [%d:%d]" % self.limit_to_frames)

        if self.rec_graph is not None:
            self.show_graph(
                self.rec_graph)

        if self.gt_graph is not None:
            self.show_graph(
                self.gt_graph)

    def show_graph(
            self,
            graph):

        logger.debug("Adding %d node meshes...", graph.number_of_nodes())

        for node, data in graph.nodes(data=True):
            self.__draw_node(node, data)

        logger.debug("Adding %d edge lines...", graph.number_of_edges())

        for u, v, data in graph.edges(data=True):
            self.__draw_edge(graph, u, v, data)

    def __block_until_closed(self):

        self.app.exec_()
        del self.viewer
        self.viewer = None

    def __propagate_selected_attributes(self, graph):

        for u, v, data in graph.edges(data=True):

            selected = self.selected_key in data and data[self.selected_key]
            selected_u = graph.nodes[u].get(self.selected_key, False)
            selected_v = graph.nodes[v].get(self.selected_key, False)
            graph.nodes[u][self.selected_key] = selected or selected_u
            graph.nodes[v][self.selected_key] = selected or selected_v

        for n, data in graph.nodes(data=True):

            if self.selected_key not in data:
                data[self.selected_key] = False

    def __compute_spimagine_pos(self, graph):

        for node, data in graph.nodes(data=True):
            if 'z' in data:
                data['spimagine_pos'] = self.__to_spimagine_coords(
                    daisy.Coordinate((data[d] for d in ['z', 'y', 'x'])))

    def __label_tracks(self, graph):

        g = nx.Graph()
        g.add_nodes_from(graph)
        g.add_edges_from(graph.edges(data=True))
        delete_edges = []
        for u, v, data in g.edges(data=True):
            selected = self.selected_key in data and data[self.selected_key]
            if not selected:
                delete_edges.append((u, v))
        g.remove_edges_from(delete_edges)

        components = nx.connected_components(g)

        for i, component in enumerate(components):
            for node in component:
                graph.nodes[node]['track'] = i + 1

    def __create_viewer(self):

        raw_data = self.raw.to_ndarray(roi=self.roi, fill_value=0)

        if len(raw_data.shape) == 5:
            raw_data = raw_data[self.channel]

        viewer = spimagine.volshow(
            raw_data,
            stackUnits=self.raw.voxel_size[1:][::-1])
        viewer.set_colormap("grays")

        viewer.glWidget.transform._transformChanged.connect(
            lambda: self.__on_transform_changed())

        return viewer

    def __draw_node(self, node, node_data):

        if 'spimagine_pos' not in node_data:
            return

        if (
                self.limit_to_frames and
                node_data['t'] not in range(
                    self.limit_to_frames[0],
                    self.limit_to_frames[1])):
            return

        if self.hide_non_selected:
            if self.selected_key in node_data:
                if not node_data[self.selected_key]:
                    return

        center = node_data['spimagine_pos']
        color = self.__get_node_color(node_data)
        radius = self.__get_node_radius(node_data)
        alpha = self.__get_node_alpha(node_data)

        self.viewer.glWidget.add_mesh(
            spimagine.gui.mesh.SphericalMesh(
                pos=center,
                r=radius,
                facecolor=color,
                alpha=alpha))

        text = self.__get_node_text(node, node_data)

        if text != "":
            self.viewer.glWidget.add_text(
                spimagine.gui.text.Text(
                    text,
                    pos=center,
                    color=color))

    def __draw_edge(self, graph, u, v, edge_data):

        node_data_u = graph.nodes[u]
        node_data_v = graph.nodes[v]

        if (
                'spimagine_pos' not in node_data_u or
                'spimagine_pos' not in node_data_v):
            return

        if (
            self.limit_to_frames and
            node_data_u['t'] not in range(
                self.limit_to_frames[0],
                self.limit_to_frames[1])):
            return

        if self.hide_non_selected:
            if self.selected_key in edge_data:
                if not edge_data[self.selected_key]:
                    return

        center_u = node_data_u['spimagine_pos']
        center_v = node_data_v['spimagine_pos']

        width = self.__get_edge_width(edge_data)
        color = self.__get_edge_color(edge_data)
        alpha = self.__get_edge_alpha(edge_data)

        self.viewer.glWidget.add_lines(
            spimagine.gui.lines.Lines(
                [center_u, center_v],
                width=width,
                linecolor=color,
                alpha=alpha))

        text = self.__get_edge_text(edge_data)

        if text != "":
            self.viewer.glWidget.add_text(
                spimagine.gui.text.Text(
                    text,
                    pos=(center_u + center_v)*0.5))

    def __on_transform_changed(self):

        t = self.viewer.glWidget.transform.dataPos
        if t == self.prev_t:
            return
        self.prev_t = t
        self.t = t + self.roi.get_begin()[0]

        logger.debug("t changed to %d" % self.t)

        self.clear_annotations()
        self.draw_annotations()

    def __to_spimagine_coords(self, coordinate):

        coordinate = np.array(coordinate, dtype=np.float32)

        # relative to ROI begin
        coordinate -= self.roi.get_begin()[1:]
        # relative to ROI size in [0, 1]
        coordinate /= np.array(self.roi.get_shape()[1:], dtype=np.float32)
        # relative to ROI size in [-1, 1]
        coordinate = coordinate*2 - 1
        # to xyz
        return coordinate[::-1]

    def __get_node_color(self, node_data):

        if self.show_track_colors and 'track' in node_data:
            return self.__id_to_color(node_data['track'])

        if self.selected_key in node_data:
            if node_data[self.selected_key]:
                return self.selected_color
            else:
                return self.nonselected_color

        return self.default_color

    def __get_node_radius(self, node_data):

        if node_data['t'] == self.t:
            return 0.04
        return 0.01

    def __get_node_alpha(self, node_data):

        if node_data['t'] == self.t:
            return 0.9
        return 0.1

    def __get_node_text(self, node, node_data):

        if node_data['t'] != self.t:
            return ""

        text = []

        if self.show_node_ids:
            text.append("ID: %d" % node)

        if self.show_node_scores:
            text.append("score: %.3f" % node_data['score'])

        text.append("track: %d" % node_data['track'])

        return ", ".join(text)

    def __get_edge_width(self, edge_data):

        return 5.0

    def __get_edge_color(self, edge_data):

        if self.selected_key in edge_data:
            if edge_data[self.selected_key]:
                return self.selected_color
            else:
                return self.nonselected_color

        return self.default_color

    def __get_edge_alpha(self, edge_data):

        return 1.0

    def __get_edge_text(self, edge_data):

        text = [
            "%s: %s" % (a, str(edge_data[a]))
            for a in self.show_edge_attributes
        ]

        return ", ".join(text)

    def __id_to_color(self, id_):

        h = id_*math.sqrt(2)

        return colorsys.hsv_to_rgb(h, 1.0, 1.0)
