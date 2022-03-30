
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F

def distillation_loss(predictions, targets, temperature):
    """Compute the distillation loss (KL divergence between predictions and targets) as described in the PET paper"""
    p = F.log_softmax(predictions / temperature, dim=1)
    q = F.softmax(targets / temperature, dim=1)

    return F.kl_div(p, q, reduction='sum') / predictions.shape[0]



class CrossEntropy_acc(nn.Module):

    def __init__(self, args):
        super(CrossEntropy_acc, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, label, labels_gt=None):

        loss = self.loss(logits, label)
        model_pre = torch.argmax(logits, dim=1)
        if labels_gt != None:
            acc = torch.sum(model_pre == labels_gt) / labels_gt.shape[0]
        else:
            acc = torch.sum(model_pre == label) / label.shape[0]
        return acc, loss



