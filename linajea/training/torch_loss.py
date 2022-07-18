"""Provides a wrapper class for the losses used in the tracking model
"""
import torch

from . import torch_model


class LossWrapper(torch.nn.Module):
    """Wraps a set of torch losses and tensorboard summaries used to train
    the tracking model of Linajea
    """
    def __init__(self, config, current_step=0):
        super().__init__()
        self.config = config
        self.voxel_size = torch.nn.Parameter(
            torch.FloatTensor(self.config.train_data.voxel_size[1:]),
            requires_grad=False)
        self.current_step = torch.nn.Parameter(
            torch.tensor(float(current_step)), requires_grad=False)

        def weighted_mse_loss(inputs, target, weight):
            ws = weight.sum() * inputs.size()[0] / weight.size()[0]
            if abs(ws) <= 0.001:
                ws = 1
            return (weight * ((inputs - target) ** 2)).sum() / ws

        def weighted_mse_loss2(inputs, target, weight):
            return (weight * ((inputs - target) ** 2)).mean()

        self.pv_loss = weighted_mse_loss
        self.ci_loss = weighted_mse_loss2

        met_sum_intv = 10
        loss_sum_intv = 1
        self.summaries = {
            "loss":                            [-1, loss_sum_intv],
            "alpha":                           [-1, loss_sum_intv],
            "cell_indicator_loss":             [-1, loss_sum_intv],
            "movement_vectors_loss_cell_mask": [-1, loss_sum_intv],
            "movement_vectors_loss_maxima":    [-1, loss_sum_intv],
            "movement_vectors_loss":           [-1, loss_sum_intv],
            'cell_ind_tpr_gt':                 [-1, met_sum_intv],
            'par_vec_cos_gt':                  [-1, met_sum_intv],
            'par_vec_diff_mn_gt':              [-1, met_sum_intv],
            'par_vec_tpr_gt':                  [-1, met_sum_intv],
            'cell_ind_tpr_pred':               [-1, met_sum_intv],
            'par_vec_cos_pred':                [-1, met_sum_intv],
            'par_vec_diff_mn_pred':            [-1, met_sum_intv],
            'par_vec_tpr_pred':                [-1, met_sum_intv],
            }

    def metric_summaries(self,
                         gt_cell_center,
                         cell_indicator,
                         cell_indicator_cropped,
                         gt_movement_vectors_cropped,
                         movement_vectors_cropped,
                         maxima,
                         maxima_in_cell_mask,
                         output_shape_2):

        # ground truth cell locations
        gt_max_loc = torch.nonzero(gt_cell_center > 0.5)

        # predicted value at those locations
        tmp = cell_indicator[list(gt_max_loc.T)]
        # true positive if > 0.5
        cell_ind_tpr_gt = torch.mean((tmp > 0.5).float())

        if not self.config.model.train_only_cell_indicator:
            # crop to nms area
            gt_cell_center_cropped = torch_model.crop(
                # l=1, d, h, w
                gt_cell_center,
                # l=1, d', h', w'
                output_shape_2)
            tp_dims = [1, 2, 3, 4, 0]

            # cropped ground truth cell locations
            gt_max_loc = torch.nonzero(gt_cell_center_cropped > 0.5)

            # ground truth movement vectors at those locations
            tmp_gt_par = gt_movement_vectors_cropped.permute(*tp_dims)[
                list(gt_max_loc.T)]

            # predicted movement vectors at those locations
            tmp_par = movement_vectors_cropped.permute(*tp_dims)[
                list(gt_max_loc.T)]

            # normalize predicted movement vectors
            normalize_pred = torch.nn.functional.normalize(tmp_par, dim=1)
            # normalize ground truth movement vectors
            normalize_gt = torch.nn.functional.normalize(tmp_gt_par, dim=1)

            # cosine similarity predicted vs ground truth movement vectors
            cos_similarity = torch.sum(normalize_pred * normalize_gt,
                                       dim=1)

            # rate with cosine similarity > 0.9
            par_vec_cos_gt = torch.mean((cos_similarity > 0.9).float())

            # distance between endpoints of predicted vs gt movement vectors
            par_vec_diff = torch.linalg.vector_norm(
                (tmp_gt_par / self.voxel_size) - (tmp_par / self.voxel_size),
                dim=1)
            # mean distance
            par_vec_diff_mn_gt = torch.mean(par_vec_diff)

            # rate with distance < 1
            par_vec_tpr_gt = torch.mean((par_vec_diff < 1).float())

        # predicted cell locations
        pred_max_loc = torch.nonzero(torch.reshape(
            torch.gt(maxima * cell_indicator_cropped, 0.2), output_shape_2))

        # predicted value at those locations
        tmp = cell_indicator_cropped[list(pred_max_loc.T)]
        # assumed good if > 0.5
        cell_ind_tpr_pred = torch.mean((tmp > 0.5).float())

        if not self.config.model.train_only_cell_indicator:
            tp_dims = [1, 2, 3, 4, 0]
            # ground truth movement vectors at those locations
            tmp_gt_par = gt_movement_vectors_cropped.permute(*tp_dims)[
                list(pred_max_loc.T)]

            # predicted movement vectors at those locations
            tmp_par = movement_vectors_cropped.permute(*tp_dims)[
                list(pred_max_loc.T)]

            # normalize predicted movement vectors
            normalize_pred = torch.nn.functional.normalize(tmp_par, dim=1)

            # normalize ground truth movement vectors
            normalize_gt = torch.nn.functional.normalize(tmp_gt_par, dim=1)

            # cosine similarity predicted vs ground truth movement vectors
            cos_similarity = torch.sum(normalize_pred * normalize_gt,
                                       dim=1)

            # rate with cosine similarity > 0.9
            par_vec_cos_pred = torch.mean((cos_similarity > 0.9).float())

            # distance between endpoints of predicted vs gt movement vectors
            par_vec_diff = torch.linalg.vector_norm(
                (tmp_gt_par / self.voxel_size) - (tmp_par / self.voxel_size),
                dim=1)

            # mean distance
            par_vec_diff_mn_pred = torch.mean(par_vec_diff)

            # rate with distance < 1
            par_vec_tpr_pred = torch.mean((par_vec_diff < 1).float())

        self.summaries['cell_ind_tpr_gt'][0] = cell_ind_tpr_gt
        self.summaries['cell_ind_tpr_pred'][0] = cell_ind_tpr_pred
        if not self.config.model.train_only_cell_indicator:
            self.summaries['par_vec_cos_gt'][0] = par_vec_cos_gt
            self.summaries['par_vec_diff_mn_gt'][0] = par_vec_diff_mn_gt
            self.summaries['par_vec_tpr_gt'][0] = par_vec_tpr_gt
            self.summaries['par_vec_cos_pred'][0] = par_vec_cos_pred
            self.summaries['par_vec_diff_mn_pred'][0] = par_vec_diff_mn_pred
            self.summaries['par_vec_tpr_pred'][0] = par_vec_tpr_pred

    def forward(self, *,
                gt_cell_indicator,
                cell_indicator,
                maxima,
                gt_cell_center,
                cell_mask=None,
                gt_movement_vectors=None,
                movement_vectors=None
                ):

        output_shape_1 = cell_indicator.size()
        output_shape_2 = maxima.size()

        # raw_cropped = torch_model.crop(raw, output_shape_1)
        cell_indicator_cropped = torch_model.crop(
            # l=1, d, h, w
            cell_indicator,
            # l=1, d', h', w'
            output_shape_2)

        if not self.config.model.train_only_cell_indicator:
            cell_mask = torch.reshape(cell_mask, (1,) + output_shape_1)
            # l=1, d', h', w'
            cell_mask_cropped = torch_model.crop(
                # l=1, d, h, w
                cell_mask,
                # l=1, d', h', w'
                output_shape_2)

            # l=1, d', h', w'
            maxima_in_cell_mask = maxima.float() * cell_mask_cropped.float()
            maxima_in_cell_mask = torch.reshape(maxima_in_cell_mask,
                                                (1,) + output_shape_2)

            # c=3, l=1, d', h', w'
            movement_vectors_cropped = torch_model.crop(
                # c=3, l=1, d, h, w
                movement_vectors,
                # c=3, l=1, d', h', w'
                (3,) + output_shape_2)

            # c=3, l=1, d', h', w'
            gt_movement_vectors_cropped = torch_model.crop(
                # c=3, l=1, d, h, w
                gt_movement_vectors,
                # c=3, l=1, d', h', w'
                (3,) + output_shape_2)

            movement_vectors_loss_cell_mask = self.pv_loss(
                # c=3, l=1, d, h, w
                gt_movement_vectors,
                # c=3, l=1, d, h, w
                movement_vectors,
                # c=1, l=1, d, h, w (broadcastable)
                cell_mask)

            # cropped
            movement_vectors_loss_maxima = self.pv_loss(
                # c=3, l=1, d', h', w'
                gt_movement_vectors_cropped,
                # c=3, l=1, d', h', w'
                movement_vectors_cropped,
                # c=1, l=1, d', h', w' (broadcastable)
                torch.reshape(maxima_in_cell_mask, (1,) + output_shape_2))
        else:
            movement_vectors_cropped = None
            gt_movement_vectors_cropped = None
            maxima_in_cell_mask = None

        # non-cropped
        if self.config.model.cell_indicator_weighted:
            if isinstance(self.config.model.cell_indicator_weighted, bool):
                self.config.model.cell_indicator_weighted = 0.00001
            cond = gt_cell_indicator < self.config.model.cell_indicator_cutoff
            weight = torch.where(cond,
                                 self.config.model.cell_indicator_weighted,
                                 1.0)
        else:
            weight = torch.tensor(1.0)

        cell_indicator_loss = self.ci_loss(
            # l=1, d, h, w
            gt_cell_indicator,
            # l=1, d, h, w
            cell_indicator,
            # l=1, d, h, w
            # cell_mask)
            weight)

        if self.config.model.train_only_cell_indicator:
            loss = cell_indicator_loss
            movement_vectors_loss = 0
        else:
            if self.config.train.movement_vectors_loss_transition_offset:
                # smooth transition from training movement vectors on complete
                # cell mask to only on maxima
                # https://www.wolframalpha.com/input/?
                # i=1.0%2F(1.0+%2B+exp(0.01*(-x%2B20000)))+x%3D0+to+40000
                f = self.config.train.movement_vectors_loss_transition_factor
                o = self.config.train.movement_vectors_loss_transition_offset
                alpha = (1.0 / (1.0 + torch.exp(f * (-self.current_step + o))))
                self.summaries['alpha'][0] = alpha

                movement_vectors_loss = (
                    movement_vectors_loss_maxima * alpha +
                    movement_vectors_loss_cell_mask * (1.0 - alpha)
                )
            else:
                movement_vectors_loss = movement_vectors_loss_cell_mask

            loss = cell_indicator_loss + movement_vectors_loss
            self.summaries['movement_vectors_loss_cell_mask'][0] = \
                movement_vectors_loss_cell_mask
            self.summaries['movement_vectors_loss_maxima'][0] = \
                movement_vectors_loss_maxima
            self.summaries['movement_vectors_loss'][0] = movement_vectors_loss

        self.summaries['loss'][0] = loss
        self.summaries['cell_indicator_loss'][0] = cell_indicator_loss

        self.metric_summaries(
            gt_cell_center,
            cell_indicator,
            cell_indicator_cropped,
            gt_movement_vectors_cropped,
            movement_vectors_cropped,
            maxima,
            maxima_in_cell_mask,
            output_shape_2)

        self.current_step += 1
        return loss, cell_indicator_loss, movement_vectors_loss, \
            self.summaries, torch.sum(cell_indicator_cropped)
