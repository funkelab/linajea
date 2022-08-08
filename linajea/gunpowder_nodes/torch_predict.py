"""Provides an adapted version of the gunpowder torch Predict node
"""
import logging
from typing import Dict, Union

from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import torch
from gunpowder.torch.nodes import Predict

logger = logging.getLogger(__name__)


class TorchPredictExt(Predict):
    """extended Torch implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        model (subclass of ``torch.nn.Module``):

            The model to use for prediction.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors (argument names of the
            ``forward`` method) in the model to array keys.

        outputs (``dict``, ``string`` or ``int`` -> :class:`ArrayKey`):

            Dictionary from the names of tensors in the network to array
            keys. If the key is a string, the tensor will be retrieved
            by checking the model for an attribute with the key as its name.
            If the key is an integer, it is interpreted as a tuple index of
            the outputs of the network.
            New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (``outputs``). This is
            useful to set the ``voxel_size``, for example, if they differ from
            the voxel size of the input arrays. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint: (``string``, optional):

            An optional path to the saved parameters for your torch module.
            These will be loaded and used for prediction if provided.

        device (``string``, optional):

            Which device to use for prediction (``"cpu"`` or ``"cuda"``).
            Default is ``"cuda"``, which falls back to CPU if CUDA is not
            available.

        spawn_subprocess (bool, optional): Whether to run ``predict`` in a
            separate process. Default is false.
    """

    def __init__(
        self,
        model,
        inputs: Dict[str, ArrayKey],
        outputs: Dict[Union[str, int], ArrayKey],
        array_specs: Dict[ArrayKey, ArraySpec] = None,
        checkpoint: str = None,
        use_swa=False,
        device="cuda",
        spawn_subprocess=False
    ):

        super(TorchPredictExt, self).__init__(
            model,
            inputs,
            outputs,
            array_specs=array_specs,
            checkpoint=checkpoint,
            device=device,
            spawn_subprocess=spawn_subprocess)

        self.use_swa = use_swa

    def start(self):

        self.use_cuda = (
            torch.cuda.is_available() and
            self.device_string == "cuda")
        logger.info(f"Predicting on {'gpu' if self.use_cuda else 'cpu'}")
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info("Device used: %s", self.device)

        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to move model to device. If you are using a child process "
                "to run your model, maybe you already initialized CUDA by sending "
                "your model to device in the main process."
            ) from e

        if self.checkpoint is not None:
            checkpoint = torch.load(self.checkpoint, map_location=self.device)
            if self.use_swa:
                logger.info("loading swa checkpoint")
                self.model = torch.optim.swa_utils.AveragedModel(self.model)
                self.model.load_state_dict(checkpoint["swa_model_state_dict"])
            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict()