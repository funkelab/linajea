from gunpowder import BatchFilter, Array
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ShuffleChannels(BatchFilter):

    def __init__(
            self,
            source):
        self.source = source

    def process(self, batch, request):
        source_request = request[self.source]
        logger.debug("Shuffling channels for %s and roi %s"
                     % (self.source, source_request.roi))
        data = batch[self.source].crop(source_request.roi).data
        np.random.shuffle(data)

        spec = self.spec[self.source]
        spec.roi = source_request.roi
        batch[self.source] = Array(data, spec)
