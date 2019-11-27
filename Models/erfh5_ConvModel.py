import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d


class erfh5_Conv3d(nn.Module):
    def __init__(self, sequence_len):
        super(erfh5_Conv3d, self).__init__()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv3d(1, 32, (17, 17, 17), padding=8)
        self.conv2 = nn.Conv3d(1, 64, (9, 9, 9), padding=4)
        self.conv3 = nn.Conv3d(64, 128, (5, 5, 5), padding=2)
        self.conv_f = nn.Conv3d(128, 1, (3, 3, 3), padding=1)

        self.conv_end = nn.Conv2d(sequence_len, 1, (3, 3), padding=1)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        # out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.dropout(out)
        out = F.relu(self.conv_f(out))

        out = self.dropout(out)
        out = torch.squeeze(out, 1)

        out = self.conv_end(out)

        out = torch.squeeze(out, 1)

        return out


class SensorToDryspotBoolModel(nn.Module):
    def __init__(self):
        super(SensorToDryspotBoolModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, (7, 7))
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 256, (3, 3))

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc_f = nn.Linear(128, 1)

    def forward(self, x):
        out = x.reshape((-1, 1, 38, 30))
        out = self.dropout(out)
        out = F.relu(self.conv1(out))
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv4(out))
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.view(out.size(0), 256, -1)

        out = out.sum(2)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc_f(out)

        return out


class erfh5_Conv2dPercentage(nn.Module):
    def __init__(self):
        super(erfh5_Conv2dPercentage, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, (15, 15))
        self.conv2 = nn.Conv2d(32, 64, (7, 7))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 256, (3, 3))

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc_f = nn.Linear(128, 1)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.dropout(out)
        out = F.relu(self.conv1(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv4(out))
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.view(out.size(0), 256, -1)

        out = out.sum(2)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc_f(out)

        return out


class erfh5_Conv25D_Frame(nn.Module):
    def __init__(self, sequence_len):
        super(erfh5_Conv25D_Frame, self).__init__()
        self.conv1 = nn.Conv2d(sequence_len, 32, (15, 15), padding=7)
        self.conv2 = nn.Conv2d(32, 64, (7, 7), padding=3)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 1, (3, 3), padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.dropout(x)
        out = F.relu(self.conv1(out))

        out = self.dropout(out)
        out = F.relu(self.conv2(out))

        out = self.dropout(out)
        out = F.relu(self.conv3(out))

        out = self.dropout(out)
        out = F.relu(self.conv4(out))
        out = torch.squeeze(out, 1)

        return out


class DrySpotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 128, 13, stride=1, padding=0)
        self.conv2 = Conv2d(128, 512, 7, stride=1, padding=0)
        self.conv3 = Conv2d(512, 2048, 5, stride=1, padding=0)
        self.conv4 = Conv2d(2048, 4096, 3, padding=0)
        self.fc_f1 = nn.Linear(4096, 1028)
        self.fc_f2 = nn.Linear(1028, 512)
        self.fc_f3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        a = x.reshape(-1, 1, 143, 111)
        b = F.relu(F.max_pool2d(self.conv1(a), kernel_size=2, stride=2))
        c = F.relu(F.max_pool2d(self.conv2(b), kernel_size=2, stride=2))
        d = F.relu(F.max_pool2d(self.conv3(c), kernel_size=2, stride=2))
        e = F.relu(self.conv4(d))
        f = e.view(e.shape[0], e.shape[1], -1).mean(2)
        f = self.dropout(f)
        g = F.relu(self.fc_f1(f))
        g = self.dropout(g)
        h = F.relu(self.fc_f2(g))
        h = self.dropout(h)
        i = torch.sigmoid(self.fc_f3(h))

        return i
