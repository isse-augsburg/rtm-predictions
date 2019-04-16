import torch
from torch import nn

class erfh5_Conv3d(nn.Module):
    def __init__(self, sequence_len):
        super(erfh5_Conv3d, self).__init__()
        self.conv1 = nn.Conv3d(1,32,(17,17,17), padding = 8)
        self.conv2 = nn.Conv3d(32,64,(9,9,9),padding = 4)
        self.conv3 = nn.Conv3d(64,128,(5,5,5),padding = 2)
        self.conv_f = nn.Conv3d(128,1,(3,3,3),padding = 1)

        self.conv_end = nn.Conv2d(sequence_len, 1, (3,3), padding = 1)

    def forward(self, x):
        out = torch.unsqueeze(x,1)
        out = self.conv1(out)

        out = self.conv2(out)
        
        out = self.conv3(out)
   
        out = self.conv_f(out)
       

        out = torch.squeeze(out,1)
       
        out = self.conv_end(out)
       
        out = torch.squeeze(out,1)
       
        return out

