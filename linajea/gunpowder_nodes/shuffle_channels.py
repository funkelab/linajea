"""Provides a gunpowder node to shuffle the order of input channels
"""
import logging

import numpy as np

from gunpowder import BatchFilter, Array

logger = logging.getLogger(__name__)


class ShuffleChannels(BatchFilter):
    """Gunpowder node to shuffle input channels

    Attributes
    ----------
    source: gp.ArrayKey
        array whose channels should be shuffled, expects channels along
        first axis
    """
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
