"""Provides a zoom/scale augment gunpowder node
"""
import logging
import random

import numpy as np
import scipy.ndimage

from gunpowder.batch_request import BatchRequest
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate

from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class ZoomAugment(BatchFilter):
    '''Zooms in or out. Useful if object size varies
    (e.g. nuclei in C.elegans embryos)

    Args:

        factor (``float``):

            Zoom factor

    '''

    def __init__(self, factor_min=1.0, factor_max=1.0,
                 spatial_dims=3, order={}, **kwargs):
        self.factor_min = factor_min
        self.factor_max_amount = factor_max - factor_min
        self.spatial_dims = spatial_dims
        self.order = order

        self.kwargs = kwargs

        logger.info("using zoom/scale augment %s %s %s %s",
                    factor_min, factor_max, order, kwargs)

    def prepare(self, request):
        self.voxel_size = self.__get_common_voxel_size(request)

        self.factor = random.random() * self.factor_max_amount + \
            self.factor_min
        logger.debug("zoom factor %s", self.factor)
        deps = BatchRequest()
        for key, spec in request.items():
            spec = spec.copy()

            if spec.roi is None:
                continue

            roi = Roi(
                spec.roi.get_begin()[-self.spatial_dims:],
                spec.roi.get_shape()[-self.spatial_dims:],
            )
            logger.debug("downstream request spatial ROI for %s is %s",
                         key, roi)

            # TODO use inverted factor? zooming in (factor>1) requires smaller
            # roi, zooming out larger
            shape = list(roi.get_shape())
            grow_factor = self.spatial_dims * [1.0/self.factor-1.0]
            grow_amount = Coordinate(
                tuple([s*gf/2 for s, gf in zip(shape, grow_factor)]))
            logger.debug("zoom grow %s %s %s", shape, grow_factor, grow_amount)
            roi = roi.grow(grow_amount, grow_amount)
            # make sure the target ROI aligns with the voxel grid (which might
            # not be the case for points)
            roi = roi.snap_to_grid(self.voxel_size[-self.spatial_dims:],
                                   mode="grow")
            logger.debug(
                "downstream request spatial ROI aligned with voxel grid for %s"
                " is %s",
                key,
                roi,
            )
            spec.roi = Roi(
                spec.roi.get_begin()[: -self.spatial_dims]
                + roi.get_begin()[-self.spatial_dims:],
                spec.roi.get_shape()[: -self.spatial_dims]
                + roi.get_shape()[-self.spatial_dims:],
            )
            deps[key] = spec

        return deps

    def process(self, batch, request):
        logger.debug("processing zoom")

        for (array_key, array) in batch.arrays.items():

            zoom = len(array.data.shape)*[1]

            recomputed_factor = (request[array_key].roi.get_shape()[-1] /
                                 array.data.shape[-1])
            logger.debug("recomputed factor array %s %s",
                         array_key, recomputed_factor)
            factor = recomputed_factor

            zoom[-self.spatial_dims:] = self.spatial_dims * [factor]
            logger.debug("zoom %s %s", array.data.shape, zoom)
            array.data = scipy.ndimage.zoom(array.data, zoom,
                                            order=self.order.get(array_key, 1))

            logger.debug("zoom %s %s",
                         array.data.shape, request[array_key].roi)
            # restore original ROIs
            array.spec.roi = request[array_key].roi

        for (graph_key, graph) in batch.graphs.items():

            recomputed_factor = (request[graph_key].roi.get_shape()[-1] /
                                 graph.spec.roi.get_shape()[-1])
            logger.debug("recomputed factor graph %s %s",
                         graph_key, recomputed_factor)
            factor = recomputed_factor

            center = graph.spec.roi.get_center()
            for node in graph.nodes:
                logger.debug("location %s %s", node.location,
                             type(node.location))
                rel_location = node.location - center
                logger.debug("relative to upstream ROI center: %s",
                             rel_location)
                rel_location[-self.spatial_dims:] = (
                    rel_location[-self.spatial_dims:] * factor)

                node.location = (np.array(center) + rel_location).astype(
                    np.float32)
                logger.debug("final location: %s", node.location)

            graph.spec.roi = request[graph_key].roi

    def __get_common_voxel_size(self, request):

        voxel_size = None
        prev = None
        for array_key in request.array_specs.keys():
            vs = self.spec[array_key].voxel_size[-self.spatial_dims:]
            if voxel_size is None:
                voxel_size = vs
            elif self.spec[array_key].voxel_size is not None:
                assert voxel_size == vs, \
                    "ElasticAugment can only be used with arrays of same " \
                    "voxel sizes, but %s has %s, and %s has %s." % (
                        array_key, self.spec[array_key].voxel_size,
                        prev, self.spec[prev].voxel_size)
            prev = array_key

        if voxel_size is None:
            raise RuntimeError("voxel size must not be None")

        return voxel_size
