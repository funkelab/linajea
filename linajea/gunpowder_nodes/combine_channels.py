from gunpowder import BatchFilter, ArraySpec, Array
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CombineChannels(BatchFilter):

    def __init__(
            self,
            channel_1,
            channel_2,
            output,
            transpose=False):
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.output = output
        self.transpose = transpose

    def setup(self):
        spec_1 = self.spec[self.channel_1]
        spec_2 = self.spec[self.channel_2]
        assert spec_1.voxel_size == spec_2.voxel_size,\
            "Channels must have same voxel size"
        roi = spec_1.roi.intersect(spec_2.roi)
        spec = ArraySpec()
        spec.roi = roi
        spec.voxel_size = spec_1.voxel_size
        self.provides(
                self.output,
                spec)

    def prepare(self, request):
        out_request = request[self.output]

        if self.channel_1 not in request:
            request[self.channel_1] = out_request.copy()
        else:
            request[self.channel_1] = request[
                    self.channel_1].union(out_request)

        if self.channel_2 not in request:
            request[self.channel_2] = out_request.copy()
        else:
            request[self.channel_2] = request[
                    self.channel_2].union(out_request)

    def process(self, batch, request):
        out_request = request[self.output]
        data_1 = batch.arrays[self.channel_1].crop(out_request.roi).data
        data_2 = batch.arrays[self.channel_2].crop(out_request.roi).data

        data = np.stack([data_1, data_2], axis=0)
        if self.transpose:
            np.random.shuffle(data)

        spec = self.spec[self.output]
        spec.roi = out_request.roi
        logger.debug("Adding key %s with spec %s to batch"
                     % (self.output, spec))
        batch[self.output] = Array(data, spec)
