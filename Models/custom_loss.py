import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def focal_loss(p, y): 
    gamma = 1


    if y == 1.0: 
        pt = p
    else:
        pt = 1.0 - p 

    ce = -1 * np.log(pt)
    fl = (1.0 - pt) ** gamma * ce
    
    return fl 

class FocalLoss(nn.Module):
    def __init__(self, gamma=1, reduction='sum', pt_epsilon=0.0001):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pt_epsilon = pt_epsilon


    def forward(self, p, y): 
        p = p[:,0]
        y = y[:,0]
        pt = torch.where(y==1.0, p, 1-p)

        ce = -1 * torch.log(pt + self.pt_epsilon)
        fl = (1 - pt)**self.gamma * ce 

        if self.reduction == 'sum':
            fl = fl.sum()
        else:
            raise "Wrong reduction chosen!"
        
        return fl

    def __str__(self): 
        string = "Focal Loss with gamma = " + str(self.gamma) + " and reduction = " + str(self.reduction)
        return string


if __name__ == "__main__":
    #out = np.array([[0.9, 0.1], [0.01, 0.99]])
    out = np.array([[0.85, 0.15], [0.99, 0.01]])
    #label = np.array([[1.0, 0.0], [1.0, 0.0]])
    label = np.array([[1.0, 0.0], [0.0, 1.0]])

    out = torch.from_numpy(out)
    label = torch.from_numpy(label)

    focalloss = FocalLoss()
    #focalloss = focal_loss(0.85, 1.0)
    fl = focalloss(out, label)



    print(fl)
    
    

