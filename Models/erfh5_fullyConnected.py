import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.data_utils import reshape_to_indeces
from Utils.training_utils import count_parameters


class SensorDryspotModelFC(nn.Module):
    def __init__(self, input_dim=1140):
        super(SensorDryspotModelFC, self).__init__()
        self.fc = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 128)
        self.fc7 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.4)

    def forward(self, inpt):
        out = F.relu(self.fc(inpt))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.drop(out)
        out = F.relu(self.fc3(out))
        out = self.drop(out)
        out = F.relu(self.fc4(out))
        out = self.drop(out)
        out = F.relu(self.fc5(out))
        out = self.drop(out)
        out = F.relu(self.fc6(out))
        out = self.drop(out)
        out = F.sigmoid(self.fc7(out))
        return out


class S1140DryspotModelFCWide(nn.Module):
    def __init__(self, input_dim=1140):
        super(S1140DryspotModelFCWide, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, _input):
        out = F.relu(self.fc(_input))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out


# self.fc = nn.Linear(input_dim, 1024)
# self.fc2 = nn.Linear(1024, 256)
# self.fc3 = nn.Linear(256, 1)
class S20DryspotModelFCWide(nn.Module):
    def __init__(self, input_dim=20, demo_mode=False):
        super(S20DryspotModelFCWide, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1)

        self.demo_mode = demo_mode

    def forward(self, _input):
        if self.demo_mode:
            _input = reshape_to_indeces(_input, ((1, 8), (1, 8)), 20)
            _input = _input.reshape(-1, 20)
        out = F.leaky_relu(self.fc(_input))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out


class S80DryspotModelFCWide(nn.Module):
    def __init__(self, input_dim=80, demo_mode=False):
        super(S80DryspotModelFCWide, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1)

        self.demo_mode = demo_mode

    def forward(self, _input):
        # With the following hack, we can use the full data here too -> saving online storage
        if self.demo_mode:
            _input = reshape_to_indeces(_input, ((1, 4), (1, 4)), 80)
            _input = _input.reshape(-1, 80)
        out = F.leaky_relu(self.fc(_input))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out


class S20DSModelFC4ChannelsFlatten(nn.Module):
    def __init__(self, input_dim=20 * 4):
        """
        This Model makes not sense, it was originally meant to be used as a baseline against the
        20 Pressuresensor network and the 20 Sensor FF/Vel. Network. But flattening the channels makes no sense,
        because the channels are meant to preserve the local information at certain time stamps.
        """
        super(S20DSModelFC4ChannelsFlatten, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, _input):
        _input = _input.view((-1, _input.shape[1] * _input.shape[2]))
        out = F.leaky_relu(self.fc(_input))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out


if __name__ == "__main__":
    model = S20DSModelFC4ChannelsFlatten()
    m = model.cuda()
    print('param count:', count_parameters(model))
    # em = torch.empty((1, 1140)).cuda()
    em = torch.empty((64, 20, 4)).cuda()
    out = m(em)

    print(out.shape)
