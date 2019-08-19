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

        self.ct1 = ConvTranspose2d(1, 8, 3, stride=2, padding=1)  # 39
        self.ct2 = ConvTranspose2d(8, 32, 5, stride=2, padding=2)  # 77
        self.ct3 = ConvTranspose2d(32, 64, 7, stride=2, padding=3)  # 233,297

        self.shaper = Conv2d(64, 64, 7, padding=3)  # 153
        self.med = Conv2d(64, 32, 5, padding=2)  # 153
        self.details = Conv2d(32, 16, 3)  # 151
        self.details2 = Conv2d(16, 1, 3, stride=2, padding=1)  # 151

    def forward(self, inputs):
        f = F.relu(self.fc(inputs))

        fr = f.reshape((-1, 1, 38, 30))

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))

        t1 = F.relu(self.shaper(k3))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))
        t4 = torch.sigmoid(self.details2(t3))
        return torch.squeeze(t4, dim=1)

