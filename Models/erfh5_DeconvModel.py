import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose2d, Conv2d, Linear

inputs = torch.randn(1, 360)


# inputs = torch.randn(1,1,20,20)


class DeconvModel_efficient(nn.Module):
    def __init__(self):
        super(DeconvModel_efficient, self).__init__()

        self.ct1 = ConvTranspose2d(1, 128, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(128,64, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(64, 32, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(32, 8, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(8, 16, 17, stride=2, padding=0)
        self.shaper = Conv2d(16, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        self.details2 = Conv2d(32, 1, 3, padding=0)

    def forward(self, inputs):
        f = inputs
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


class DeconvModel(nn.Module):
    def __init__(self, input_dim=1140):
        super(DeconvModel, self).__init__()
        self.fc = Linear(input_dim, 1140)

        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(16, 32, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(32, 64, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(64, 128, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(128, 64, 17, stride=2, padding=0)
        self.shaper = Conv2d(64, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        self.details2 = Conv2d(32, 1, 3, padding=0)

    def forward(self, inputs):
        f = inputs
        # f = F.relu(self.fc(inputs))

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


class DeconvModel2x(nn.Module):
    def __init__(self, input_dim=1140):
        super(DeconvModel2x, self).__init__()

        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(16, 32, 5, stride=2, padding=2)
        self.ct3 = ConvTranspose2d(32, 32, 7, stride=2, padding=3)
        self.ct4 = ConvTranspose2d(32, 64, 15, stride=2, padding=0)
        self.ct5 = ConvTranspose2d(64, 128, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(128, 64, 17, stride=2, padding=0)
        self.shaper = Conv2d(64, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        self.details2 = Conv2d(32, 1, 3, padding=0)

    def forward(self, inputs):
        f = inputs
        # f = F.relu(self.fc(inputs))

        fr = f.reshape((-1, 1, 38, 30))
        fr = fr[:, :, ::2, ::2]

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k4 = F.relu(self.ct4(k3))
        k5 = F.relu(self.ct5(k4))

        t1 = F.relu(self.shaper0(k5))
        t1 = F.relu(self.shaper(t1))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))
        t4 = torch.sigmoid(self.details2(t3))
        return torch.squeeze(t4, dim=1)


class DeconvModel4x(nn.Module):
    def __init__(self, input_dim=1140):
        super(DeconvModel4x, self).__init__()

        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(16, 32, 5, stride=2, padding=2)
        self.ct3 = ConvTranspose2d(32, 32, 7, stride=2, padding=3)
        self.ct4 = ConvTranspose2d(32, 32, 9, stride=2, padding=4)
        self.ct5 = ConvTranspose2d(32, 64, 15, stride=2, padding=0)
        self.ct6 = ConvTranspose2d(64, 128, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(128, 64, 17, stride=2, padding=0)
        self.shaper = Conv2d(64, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        self.details2 = Conv2d(32, 1, 3, padding=0)

    def forward(self, inputs):
        f = inputs

        fr = f.reshape((-1, 1, 38, 30))
        fr = fr[:, :, 2::4, 2::4]  # len = 63

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k4 = F.relu(self.ct4(k3))
        k5 = F.relu(self.ct5(k4))
        k6 = F.relu(self.ct6(k5))

        t1 = F.relu(self.shaper0(k6))
        t1 = F.relu(self.shaper(t1))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))
        t4 = torch.sigmoid(self.details2(t3))
        return torch.squeeze(t4, dim=1)


class DeconvModel8x(nn.Module):
    def __init__(self):
        super(DeconvModel8x, self).__init__()

        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(16, 32, 5, stride=2, padding=2)
        self.ct3 = ConvTranspose2d(32, 32, 7, stride=2, padding=3)
        self.ct4 = ConvTranspose2d(32, 32, 9, stride=2, padding=4)
        self.ct5 = ConvTranspose2d(32, 32, 12, stride=2, padding=4)
        self.ct6 = ConvTranspose2d(32, 64, 15, stride=2, padding=0)
        self.ct7 = ConvTranspose2d(64, 128, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(128, 64, 17, stride=2, padding=0)
        self.shaper = Conv2d(64, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        self.details2 = Conv2d(32, 1, 3, padding=0)

    def forward(self, inputs):
        f = inputs

        fr = f.reshape((-1, 1, 38, 30))
        fr = fr[:, :, ::8, ::8]  # len = 63

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k4 = F.relu(self.ct4(k3))
        k5 = F.relu(self.ct5(k4))
        k6 = F.relu(self.ct6(k5))
        k7 = F.relu(self.ct7(k6))

        t1 = F.relu(self.shaper0(k7))
        t1 = F.relu(self.shaper(t1))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))
        t4 = torch.sigmoid(self.details2(t3))
        return torch.squeeze(t4, dim=1)
