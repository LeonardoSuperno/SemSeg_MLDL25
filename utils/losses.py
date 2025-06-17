import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss without alpha weighting.
    Focal Loss = - (1 - pt)^gamma * log(pt)
    
    :param num_class: number of classes
    :param gamma: focusing parameter for modulating factor (1-pt)
    :param size_average: whether to average the loss over batch
    :param ignore_label: label to ignore in target
    """

    def __init__(self, num_class, gamma=2.0, size_average=True, ignore_label=255):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6
        self.ignore_label = ignore_label

    def forward(self, logit, target):
        logit = F.softmax(logit, dim=1)
        n, c, h, w = logit.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        logit = logit.transpose(1, 2).transpose(2, 3).contiguous()
        logit = logit[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target = target.view(-1, 1)

        pt = logit.gather(1, target).view(-1) + self.eps 
        logpt = pt.log()

        loss = -1 * torch.pow(1.0 - pt, self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index):
        super().__init__()
        self.epsilon = 1e-5
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, output, target):
        if self.ignore_index is not None:
            target = target.clone()
            target[target == self.ignore_index] = output.size()[1]
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)

        intersection = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice_score = 2. * intersection / (denominator + self.epsilon)
        return torch.mean(1. - dice_score)


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes+1, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target[:, :classes, ...]