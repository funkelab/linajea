from gunpowder import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.morphology import enlarge_binary_map
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AddParentVectors(BatchFilter):

    def __init__(
            self, points, array, mask, radius,
            move_radius=0, array_spec=None, dense=False,
            subsampling=None):

        self.points = points
        self.array = array
        self.mask = mask
        self.radius = np.array([radius]).flatten().astype(np.float32)
        self.move_radius = move_radius
        if array_spec is None:
            self.array_spec = ArraySpec()
        else:
            self.array_spec = array_spec

        self.dense = dense
        self.subsampling = subsampling

    def setup(self):

        points_roi = self.spec[self.points].roi

        if self.array_spec.voxel_size is None:
            self.array_spec.voxel_size = Coordinate((1,)*points_roi.dims())

        if self.array_spec.dtype is None:
            self.array_spec.dtype = np.float32

        self.array_spec.roi = points_roi.copy()
        self.provides(
            self.array,
            self.array_spec)
        self.provides(
            self.mask,
            self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):
        context = np.ceil(self.radius).astype(int)

        dims = self.array_spec.roi.dims()
        if len(context) == 1:
            context = context.repeat(dims)

        # regardless of the radius, we need at least the previous and next
        # frame to see parents
        context[0] = max(1, context[0])

        # we also need to expand the context to try and include parents
        for i in range(1, len(context)):
            context[i] = max(context[i], self.move_radius)

        logger.debug ("parent vector context %s", context)
        # request points in a larger area
        points_roi = request[self.array].roi.grow(
                Coordinate(context),
                Coordinate(context))

        # however, restrict the request to the points actually provided
        points_roi = points_roi.intersect(self.spec[self.points].roi)
        logger.debug("Requesting points in roi %s" % points_roi)
        request[self.points] = GraphSpec(roi=points_roi)

    def process(self, batch, request):

        points = batch.graphs[self.points]
        voxel_size = self.spec[self.array].voxel_size

        # get roi used for creating the new array (points_roi does not
        # necessarily align with voxel size)
        enlarged_vol_roi = points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_vol_roi.get_begin()/voxel_size
        shape = enlarged_vol_roi.get_shape()/voxel_size
        data_roi = Roi(offset, shape)

        # points ROI is at least +- 1 in t of requested array ROI, we can save
        # some time by shaving the excess off
        data_roi = data_roi.grow((-1, 0, 0, 0), (-1, 0, 0, 0))

        logger.debug("Points in %s", points.spec.roi)
        if logger.isEnabledFor(logging.DEBUG):
            for node in points.nodes:
                logger.debug("%d, %s", node.id, node.location)
            logger.debug("Data roi in voxels: %s", data_roi)
            logger.debug("Data roi in world units: %s", data_roi*voxel_size)

        parent_vectors_data, mask_data = self.__draw_parent_vectors(
            points,
            data_roi,
            voxel_size,
            enlarged_vol_roi.get_begin(),
            self.radius,
            request[self.points].roi)

        # create array and crop it to requested roi
        spec = self.spec[self.array].copy()
        spec.roi = data_roi*voxel_size
        parent_vectors = Array(
            data=parent_vectors_data,
            spec=spec)
        logger.debug("Cropping parent vectors to %s", request[self.array].roi)
        batch.arrays[self.array] = parent_vectors.crop(request[self.array].roi)

        # create mask and crop it to requested roi
        spec = self.spec[self.mask].copy()
        spec.roi = data_roi*voxel_size
        mask = Array(
            data=mask_data,
            spec=spec)
        logger.debug("Cropping mask to %s", request[self.mask].roi)
        batch.arrays[self.mask] = mask.crop(request[self.mask].roi)

        # restore requested ROI of points
        if self.points in request:
            request_roi = request[self.points].roi
            points.spec.roi = request_roi
            for point in list(points.nodes):
                if not request_roi.contains(point.location):
                    points.remove_node(point)

            if points.num_vertices() == 0:
                logger.warning("Returning empty batch for key %s and roi %s"
                               % (self.points, request_roi))

    def __draw_parent_vectors(
            self, points, data_roi, voxel_size, offset, radius, final_roi):

        # 4D: t, z, y, x
        shape = data_roi.get_shape()
        l, d, h, w = shape

        # 5D: t, c, z, y, x (c=[0, 1, 2])
        coords = np.array(
                [
                    # 4D: c, z, y, x
                    np.meshgrid(
                        np.arange(0, d),
                        np.arange(0, h),
                        np.arange(0, w),
                        indexing='ij')
                ]*l,
                dtype=np.float32)

        # 5D: c, t, z, y, x
        coords = coords.transpose((1, 0, 2, 3, 4))
        coords[0, :] *= voxel_size[1]
        coords[1, :] *= voxel_size[2]
        coords[2, :] *= voxel_size[3]
        coords[0, :] += offset[1]
        coords[1, :] += offset[2]
        coords[2, :] += offset[3]

        parent_vectors = np.zeros_like(coords)
        mask = np.zeros(shape, dtype=bool)

        logger.debug(
            "Adding parent vectors for %d points...",
            points.num_vertices())

        if points.num_vertices() == 0:
            return parent_vectors, mask.astype(np.float32)

        empty = True
        cnt = 0
        total = 0
        for point_id, point in points.data.items():
            if self.subsampling is not None and point.value > self.subsampling:
                logger.debug("skipping point %s %s due to subsampling %s",
                             point, point.value, self.subsampling)
                continue

        avg_radius = []
        for point in points.nodes:
            if point.attrs.get('value') is not None:
                r = point.attrs['value'][0]
                avg_radius.append(point.attrs['value'][0])
        avg_radius = (int(np.ceil(np.mean(avg_radius)))
                      if len(avg_radius) > 0 else radius)

        for point in points.nodes:
            # get the voxel coordinate, 'Coordinate' ensures integer
            v = Coordinate(point.location/voxel_size)

            if not data_roi.contains(v):
                logger.debug(
                    "Skipping point at %s outside of requested data ROI",
                    v)
                continue

            if point.attrs.get("parent_id") is None:
                logger.warning("Skipping point without parent")
                continue

            if not points.contains(point.attrs["parent_id"]):
                if final_roi.contains(point.location):
                    logger.warning(
                        "parent %d of %d not in %s",
                        point.attrs["parent_id"],
                        point.id, self.points)
                    logger.debug("request roi: %s" % data_roi)
                if not self.dense:
                    continue
            empty = False
            # get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            mask[v] = 1

        enlarge_binary_map(
            mask,
            avg_radius,
            voxel_size,
            in_place=True)

        mask_tmp = np.zeros(shape, dtype=bool)
        mask_tmp[shape//2] = 1
        enlarge_binary_map(
            mask_tmp,
            avg_radius,
            voxel_size,
            in_place=True)

        coords = np.argwhere(mask_tmp)
        _, z_min, y_min, x_min = coords.min(axis=0)
        _, z_max, y_max, x_max = coords.max(axis=0)
        mask_cut = mask_tmp[:,
                            z_min:z_max+1,
                            y_min:y_max+1,
                            x_min:x_max+1]
        logger.info("mask cut %s", mask_cut.shape)

        point_mask = np.zeros(shape, dtype=bool)

        for point in points.nodes:

            # get the voxel coordinate, 'Coordinate' ensures integer
            v = Coordinate(point.location/voxel_size)

            if not data_roi.contains(v):
                continue

            total += 1
            if point.attrs.get("parent_id") is None:
                continue

            if not points.contains(point.attrs["parent_id"]):
                if not self.dense:
                    continue
            empty = False
            # get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            logger.debug(
                "Rasterizing point %s at %s, shape %s",
                point.location,
                point.location/voxel_size - data_roi.get_begin(), shape)

            if not points.contains(point.attrs["parent_id"]) and self.dense:
                continue

            cnt += 1
            parent = points.node(point.attrs["parent_id"])

            s_begin = []
            s_end = []
            mc_begin = []
            mc_end = []
            for c, ms, s in zip(v, mask_cut.shape, shape):
                b = c-ms//2
                if b < 0:
                    s_begin.append(0)
                    mc_begin.append(-b)
                else:
                    s_begin.append(b)
                    mc_begin.append(0)
                e = c+ms//2+1
                if e > s:
                    s_end.append(s)
                    mc_end.append(s-e)
                else:
                    s_end.append(e)
                    mc_end.append(ms)

            slices = (slice(None),
                      slice(s_begin[1], s_end[1]),
                      slice(s_begin[2], s_end[2]),
                      slice(s_begin[3], s_end[3]))
            mc_slices = (slice(None),
                         slice(mc_begin[1], mc_end[1]),
                         slice(mc_begin[2], mc_end[2]),
                         slice(mc_begin[3], mc_end[3]))

            point_mask[:] = 0
            point_mask[slices] = mask_cut[mc_slices]

            parent_vectors[0][point_mask] = (parent.location[1]
                                             - coords[0][point_mask])
            parent_vectors[1][point_mask] = (parent.location[2]
                                             - coords[1][point_mask])
            parent_vectors[2][point_mask] = (parent.location[3]
                                             - coords[2][point_mask])

        parent_vectors[0][np.logical_not(mask)] = 0
        parent_vectors[1][np.logical_not(mask)] = 0
        parent_vectors[2][np.logical_not(mask)] = 0

        if empty:
            logger.warning("No parent vectors written for points %s"
                           % points.nodes)
        logger.info("written {}/{}".format(cnt, total))

        return parent_vectors, mask.astype(np.float32)
