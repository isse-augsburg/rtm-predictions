import torch.nn as nn
import torch.nn.functional as F  


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

