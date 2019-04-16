import torch
from torch import nn
import torch.nn.functional as F

class erfh5_Conv3d(nn.Module):
    def __init__(self, sequence_len):
        super(erfh5_Conv3d, self).__init__()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv3d(1,32,(17,17,17), padding = 8)
        self.conv2 = nn.Conv3d(1,64,(9,9,9),padding = 4)
        self.conv3 = nn.Conv3d(64,128,(5,5,5),padding = 2)
        self.conv_f = nn.Conv3d(128,1,(3,3,3),padding = 1)

        self.conv_end = nn.Conv2d(sequence_len, 1, (3,3), padding = 1)

    def forward(self, x):
        out = torch.unsqueeze(x,1)
        #out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.dropout(out)
        out = F.relu(self.conv_f(out))
       
        out = self.dropout(out)
        out = torch.squeeze(out,1)
       
        out = self.conv_end(out)
       
        out = torch.squeeze(out,1)
       
        return out

class erfh5_Conv2dPercentage(nn.Module):
    def __init__(self):
        super(erfh5_Conv2dPercentage, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1,32,(15,15))
        self.conv2 = nn.Conv2d(32,64,(7,7))
        self.conv3 = nn.Conv2d(64,128,(3,3))
        self.conv4 = nn.Conv2d(128,256,(3,3))

        self.fc1 = nn.Linear(256,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 128)
        self.fc_f = nn.Linear(128,1)

    def forward(self,x):
        out = torch.unsqueeze(x,1)
        out = self.dropout(out)
        out = F.relu(self.conv1(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out =  F.relu(self.conv2(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out =  F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out =  F.relu(self.conv4(out))
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.view(out.size(0), 256, -1)
        
        out = out.sum(2)
        
        out =  F.relu(self.fc1(out))
        out = self.dropout(out)
        out =  F.relu(self.fc2(out))
        out = self.dropout(out)
        out =  F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc_f(out)

        return out

