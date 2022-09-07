"""Provides a noise augment gunpowder node
"""
import logging

import numpy as np
import skimage

from gunpowder.batch_request import BatchRequest

from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class NoiseAugment(BatchFilter):
    '''Add random noise to an array. Uses the scikit-image function
    skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and
    additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and
            within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        seed (``int``):

            Optionally set a random seed, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by
            clipping values in the end, see
            scikit-image documentation
    '''

    def __init__(self, array, mode='gaussian', seed=None, clip=True,
                 check_val_range=True, **kwargs):
        self.array = array
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.check_val_range = check_val_range
        self.kwargs = kwargs

        logger.info("using noise augment %s %s", mode, kwargs)

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, \
            ("Noise augmentation requires float types for the raw array (not "
             + str(raw.data.dtype) + "). Consider using Normalize before.")
        if self.check_val_range:
            assert raw.data.min() >= -1 and raw.data.max() <= 1, \
                ("Noise augmentation expects raw values in [-1,1] or [0,1]."
                 "Consider using Normalize before.")

        kwargs = self.kwargs
        if self.mode == 'gaussian' or self.mode == 'speckle':
            var = self.kwargs['var']
            if isinstance(var, list):
                if len(var) == 1:
                    var = np.random.uniform(high=var[0])
                else:
                    var = np.random.uniform(low=var[0], high=var[1])
            kwargs = {'var': var}
        if self.mode == 's&p':
            amount = self.kwargs['amount']
            if isinstance(amount, list):
                if len(amount) == 1:
                    amount = np.random.uniform(high=amount[0])
                else:
                    amount = np.random.uniform(low=amount[0], high=amount[1])
            kwargs = {'amount': amount}
        raw.data = skimage.util.random_noise(
            raw.data,
            mode=self.mode,
            seed=self.seed,
            clip=self.clip,
            **kwargs).astype(raw.data.dtype)
