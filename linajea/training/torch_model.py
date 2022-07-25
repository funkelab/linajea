"""Provides a U-Net based tracking model class using torch
"""
import json
import logging
import os

import torch

from funlib.learn.torch.models import UNet, ConvPass

from .utils import (crop,
                    crop_to_factor)

logger = logging.getLogger(__name__)


class UnetModelWrapper(torch.nn.Module):
    """Wraps a torch U-Net implementation and extends it to the tracking
    model used in Linajea

    Supports multiple styles of U-Nets:
    - a single network for both cell indicator and movement vectors (single)
    - two separate networks (split)
    - a shared encoder and two decoders (multihead)

    Adds a layer to directly perform non maximum suppression using a 3d
    pooling layer with stride 1

    Input and Output shapes can be precomputed using `inout_shapes` (they
    can differ as valid padding is used by default)
    """
    def __init__(self, config, current_step=0):
        super().__init__()
        self.config = config
        self.current_step = current_step

        num_fmaps = (config.model.num_fmaps
                     if isinstance(config.model.num_fmaps, list)
                     else [config.model.num_fmaps, config.model.num_fmaps])

        if config.model.unet_style == "split" or \
           config.model.unet_style == "single" or \
           self.config.model.train_only_cell_indicator:
            num_heads = 1
        elif config.model.unet_style == "multihead":
            num_heads = 2
        else:
            raise RuntimeError("invalid unet style, should be split, single "
                               "or multihead")

        self.unet_cell_ind = UNet(
            in_channels=1,
            num_fmaps=num_fmaps[0],
            fmap_inc_factor=config.model.fmap_inc_factors,
            downsample_factors=config.model.downsample_factors,
            kernel_size_down=config.model.kernel_size_down,
            kernel_size_up=config.model.kernel_size_up,
            constant_upsample=config.model.constant_upsample,
            upsampling=config.model.upsampling,
            num_heads=num_heads,
        )

        if config.model.unet_style == "split" and \
           not self.config.model.train_only_cell_indicator:
            self.unet_mov_vec = UNet(
                in_channels=1,
                num_fmaps=num_fmaps[1],
                fmap_inc_factor=config.model.fmap_inc_factors,
                downsample_factors=config.model.downsample_factors,
                kernel_size_down=config.model.kernel_size_down,
                kernel_size_up=config.model.kernel_size_up,
                constant_upsample=config.model.constant_upsample,
                upsampling=config.model.upsampling,
                num_heads=1,
            )

        self.cell_indicator_batched = ConvPass(num_fmaps[0], 1, [[1, 1, 1]],
                                               activation='Sigmoid')
        self.movement_vectors_batched = ConvPass(num_fmaps[1], 3, [[1, 1, 1]],
                                                 activation=None)

        self.nms = torch.nn.MaxPool3d(config.model.nms_window_shape, stride=1,
                                      padding=0)

    def init_layers(self):
        # the default init in pytorch is a bit strange
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        # some modified (for backwards comp) version of kaiming
        # breaks training, cell_ind -> 0
        # activation func relu
        def init_weights(m):
            # print("init", m)
            if isinstance(m, torch.nn.Conv3d):
                # print("init")
                # torch.nn.init.xavier_uniform(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.0)

        # activation func sigmoid
        def init_weights_sig(m):
            if isinstance(m, torch.nn.Conv3d):
                # torch.nn.init.xavier_uniform(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        logger.info("initializing model..")
        self.apply(init_weights)
        self.cell_indicator_batched.apply(init_weights_sig)
        if not self.config.model.train_only_cell_indicator:
            self.movement_vectors_batched.apply(init_weights)

    def inout_shapes(self, device):
        logger.info("getting train/test output shape by running model twice")
        input_shape_predict = self.config.model.predict_input_shape
        self.eval()
        with torch.no_grad():
            trial_run_predict = self.forward(
                torch.zeros(input_shape_predict,
                            dtype=torch.float32).to(device))
        self.train()
        logger.info("test done")
        if self.config.model.train_only_cell_indicator:
            _, _, trial_max_predict = trial_run_predict
        else:
            _, _, trial_max_predict, _ = trial_run_predict
        output_shape_predict = trial_max_predict.size()
        net_config = {
            'input_shape': input_shape_predict,
            'output_shape_2': output_shape_predict
        }
        with open('test_net_config.json', 'w') as f:
            json.dump(net_config, f)

        input_shape = self.config.model.train_input_shape
        trial_run = self.forward(torch.zeros(input_shape,
                                             dtype=torch.float32).to(device))
        if self.config.model.train_only_cell_indicator:
            trial_ci, _, trial_max = trial_run
        else:
            trial_ci, _, trial_max, _ = trial_run
        output_shape_1 = trial_ci.size()
        output_shape_2 = trial_max.size()
        logger.info("train done")
        net_config = {
            'input_shape': input_shape,
            'output_shape_1': output_shape_1,
            'output_shape_2': output_shape_2
        }
        with open('train_net_config.json', 'w') as f:
            json.dump(net_config, f)
        return input_shape, output_shape_1, output_shape_2

    def forward(self, raw):
        raw = torch.reshape(raw, [1, 1] + list(raw.size()))
        model_out = self.unet_cell_ind(raw)
        if self.config.model.unet_style != "multihead" or \
           self.config.model.train_only_cell_indicator:
            model_out = [model_out]
        cell_indicator_batched = self.cell_indicator_batched(model_out[0])
        output_shape_1 = list(cell_indicator_batched.size())[1:]
        cell_indicator = torch.reshape(cell_indicator_batched, output_shape_1)

        if self.config.model.unet_style == "single":
            movement_vectors_batched = self.movement_vectors_batched(
                model_out[0])
            movement_vectors = torch.reshape(movement_vectors_batched,
                                             [3] + output_shape_1)
        elif (self.config.model.unet_style == "split" and
              not self.config.model.train_only_cell_indicator):
            model_mov_vec = self.unet_mov_vec(raw)
            movement_vectors_batched = self.movement_vectors_batched(
                model_mov_vec)
            movement_vectors = torch.reshape(movement_vectors_batched,
                                             [3] + output_shape_1)
        else:  # self.config.model.unet_style == "multihead"
            movement_vectors_batched = self.movement_vectors_batched(
                model_out[1])
            movement_vectors = torch.reshape(movement_vectors_batched,
                                             [3] + output_shape_1)

        maxima = self.nms(cell_indicator_batched)
        if not self.training:
            factor_product = None
            for factor in self.config.model.downsample_factors:
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f*ff
                        for f, ff in zip(factor, factor_product))
            maxima = crop_to_factor(
                maxima,
                factor=factor_product,
                kernel_sizes=[[1, 1, 1]])

        output_shape_2 = tuple(list(maxima.size())[1:])
        maxima = torch.reshape(maxima, output_shape_2)

        if not self.training:
            cell_indicator = crop(cell_indicator, output_shape_2)
            maxima = torch.eq(maxima, cell_indicator)
            if not self.config.model.train_only_cell_indicator:
                movement_vectors = crop(movement_vectors, output_shape_2)
        else:
            cell_indicator_cropped = crop(cell_indicator, output_shape_2)
            maxima = torch.eq(maxima, cell_indicator_cropped)

        raw_cropped = crop(raw, output_shape_2)
        raw_cropped = torch.reshape(raw_cropped, output_shape_2)
        # print(torch.min(raw), torch.max(raw))
        # print(torch.min(raw_cropped), torch.max(raw_cropped))

        if self.config.model.train_only_cell_indicator:
            return cell_indicator, maxima, raw_cropped
        else:
            return cell_indicator, maxima, raw_cropped, movement_vectors
