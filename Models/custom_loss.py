import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def focal_loss(p, y): 
    gamma = torch.tensor([2])


    if y == 1.0: 
        pt = p
    else:
        pt = 1 - p 

    ce = -torch.log(pt)
    fl = (1 - pt) ** gamma
    
    return fl 

class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, p, y): 
        logpt = F.log_softmax(p)
        
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.sum()

