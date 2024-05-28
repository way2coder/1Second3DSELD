'''
Multi criterions are encompassed in this file. 
One can design custom loss function in this file and add its clss name in run.py
'''
import math
from itertools import permutations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Tuple

class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')  # The autograd system in PyTorch automatically computes the gradients of tensors

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZD)=4, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss
    



class SELLoss(_Loss):
    def __init__(self, max_num_sources: int = 13, alpha: float = 1.0, reduction='none') -> None:
        super(SELLoss, self).__init__(reduction=reduction)
        if not (0 <= alpha <= 1):
            raise ValueError('The weighting parameter must be a number between 0 and 1.')
        self.alpha = alpha
        # self.permutations = torch.from_numpy(np.array(list(permutations(range(max_num_sources)))))
        self.max_num_sources = max_num_sources
        self.loss_dictionary = {'l1':[], 'l2':[], 'l3':[]}

    @staticmethod
    def compute_spherical_distance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        sine_term = torch.sin(y_pred[..., 0]) * torch.sin(y_true[..., 0])
        cosine_term = torch.cos(y_pred[..., 0]) * torch.cos(y_true[..., 0]) * torch.cos(y_true[..., 1] - y_pred[..., 1])
        return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # breakpoint()
        targets = targets.view(targets.shape[0], targets.shape[1], self.max_num_sources, 4)
        source_activity_pred, direction_of_arrival_pred= predictions[..., 0], predictions[..., 1:4]
        source_activity_target, direction_of_arrival_target = targets[...,0], targets[...,1:4]
        '''
        source_activity_pred (batchsize T 4)  direction_of_arrival_pred(batchsize T 4 2)

        '''
        # Create mask for active sources
        source_activity_mask = source_activity_target.bool().unsqueeze(-1).expand_as(direction_of_arrival_pred)
        
        # Apply mask
        direction_of_arrival_pred_masked = direction_of_arrival_pred.masked_fill(~source_activity_mask, 0)
        direction_of_arrival_target_masked = direction_of_arrival_target.masked_fill(~source_activity_mask, 0)


        # BCE loss for source activity
        source_activity_bce_loss = F.binary_cross_entropy_with_logits(source_activity_pred, source_activity_target, reduction=self.reduction)
    
        # Spherical distance
        spherical_distance = self.compute_spherical_distance(direction_of_arrival_pred_masked[...,:-1], direction_of_arrival_target_masked[...,:-1])
        # important! the  source_activity_bce_loss and the spherical_distance's shape are all [16, 25, 4]


        distance_criterion = nn.MSELoss()
        distance_loss = distance_criterion(direction_of_arrival_pred_masked[...,-1], direction_of_arrival_target_masked[...,-1])
        l1 = torch.mean(source_activity_bce_loss)
        l2 = torch.mean(spherical_distance)
        l3 = torch.mean(distance_loss)
        self.loss_dictionary['l1'].append(l1)
        self.loss_dictionary['l2'].append(l2)
        self.loss_dictionary['l3'].append(l3)
        total_loss =l1 + l2 + l3
        
        # meta_data = {
        #     'source_activity_loss': torch.mean(source_activity_bce_loss),  # Convert to Python number
        #     'direction_of_arrival_loss': torch.mean(spherical_distance) 
        #  # Convert to Python number
        # }

        return total_loss

def compute_angular_distance(x, y):
    """Computes the angle between two spherical direction-of-arrival points.

    :param x: single direction-of-arrival, where the first column is the azimuth and second column is elevation
    :param y: single or multiple DoAs, where the first column is the azimuth and second column is elevation
    :return: angular distance
    """
    if np.ndim(x) != 1:
        raise ValueError('First DoA must be a single value.')

    return np.arccos(np.sin(x[0]) * np.sin(y[0]) + np.cos(x[0]) * np.cos(y[0]) * np.cos(y[1] - x[1]))


def get_num_params(model):
    """Returns the number of trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)