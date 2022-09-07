"""Provides a gunpowder provider node that selects one of the upstream
providers based on the current step, can be used to switch between training
and validation on the fly.
"""
import logging
import copy

from gunpowder.nodes.batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class TrainValProvider(BatchProvider):
    '''Select one of the upstream providers based on current step:

        (train + val) + TrainValProvider()

    will create a provider that relays requests to providers ``train``, or
    ``val``. Array and point keys of ``train``, ``val`` should be the same.

    Args:
        step (``int``, scalar):

            Every STEP iteration the validation provider is used instead of
            the train provider.
            (current_step % step != 1)
            This allows for interleaved validation. Tensorboard summaries are
            written to separate files, the network weights are not updated.

        init_step (``int``, scalar, optional):

            If a training checkpoint is reloaded and training continued
            from an iteration != 0.
    '''

    def __init__(self, step=50, init_step=0):
        self.step = step
        self.current_step = init_step

    def setup(self):
        self.enable_placeholders()
        assert len(self.get_upstream_providers()) == 2, (
            "two batch providers (for train and val) must be added to the "
            "TrainValProvider")

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

    def provide(self, request):
        # Random seed is set in provide rather than prepare since this node
        # is not a batch filter
        providers = self.get_upstream_providers()
        provider = providers[0] if self.current_step % self.step != 1 \
            else providers[1]
        logger.debug("upstream providers: %s, choice train? %s",
                     providers, self.current_step % self.step != 1)
        self.current_step += 1
        return provider.request_batch(request)
