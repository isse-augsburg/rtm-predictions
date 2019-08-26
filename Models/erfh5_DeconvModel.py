import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ConvTranspose2d, Conv2d, Linear

inputs = torch.randn(1, 360)


# inputs = torch.randn(1,1,20,20)

class DeconvModel(nn.Module):
    def __init__(self, input_dim=1140):
        super(DeconvModel, self).__init__()
        self.fc = Linear(input_dim,  1140)

        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=1)  
        self.ct2 = ConvTranspose2d(16, 32, 7, stride=2, padding=3)  
        self.ct3 = ConvTranspose2d(32, 64, 15, stride=2, padding=7)  
        self.ct4 = ConvTranspose2d(64, 128, 17, stride=2, padding=8)
        
        self.shaper0 = Conv2d(128, 64, 17, stride=2, padding=8)
        self.shaper = Conv2d(64, 32, 15, stride=2, padding=7)  
        self.med = Conv2d(32, 32, 7, padding=3) 
        self.details = Conv2d(32, 32, 3) 
        self.details2 = Conv2d(32, 1, 3,  padding=1) 

    def forward(self, inputs):
        f = F.relu(self.fc(inputs))

        fr = f.reshape((-1, 1, 38, 30))

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k3 = F.relu(self.ct4(k3))

        t1 = F.relu(self.shaper0(k3))
        t1 = F.relu(self.shaper(t1))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))
        t4 = torch.sigmoid(self.details2(t3))
        return torch.squeeze(t4, dim=1)

