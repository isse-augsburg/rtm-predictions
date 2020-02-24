import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_dim=20):
        super(S20DryspotModelFCWide, self).__init__()
        self.fc = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, _input):
        out = F.leaky_relu(self.fc(_input))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out


if __name__ == "__main__":
    model = S1140DryspotModelFCWide()
    m = model.cuda()
    print('param count:', count_parameters(model))
    em = torch.empty((1, 1140)).cuda()
    out = m(em)

    print(out.shape)
