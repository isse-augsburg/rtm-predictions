import os
from pathlib import Path

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowfrontFeatures_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, batch_size=16, num_layers=3):
        super(FlowfrontFeatures_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nlayers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=False, num_layers=self.nlayers,
                            bidirectional=False, dropout=0.2)
        self.hidden2hidden1 = nn.Linear(int(hidden_dim), 1024)
        self.hidden2hidden2 = nn.Linear(1024, 8192)
        self.hidden2hidden3 = nn.Linear(8192, 24025)
        self.drop = nn.Dropout(0.30)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.hidden2hidden1.bias.data.fill_(0)
        self.hidden2hidden1.weight.data.uniform_(-initrange, initrange)
        self.hidden2hidden2.bias.data.fill_(0)
        self.hidden2hidden2.weight.data.uniform_(-initrange, initrange)
        self.hidden2hidden3.bias.data.fill_(0)
        self.hidden2hidden3.weight.data.uniform_(-initrange, initrange)

    # def init_hidden(self):
    #     return [
    #         Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_dim)),
    #         Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_dim)),
    #     ]

    def forward(self, x):
        """  for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous() """

        x = x.permute(1, 0, 2)
        lstm_out, hidden = self.lstm(x)

        out = lstm_out[-1]

        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = F.relu(self.hidden2hidden1(out))
        out = self.drop(out)
        out = F.relu(self.hidden2hidden2(out))
        out = self.drop(out)
        out = F.relu(self.hidden2hidden3(out))
        out = self.drop(out)
        # out = torch.sigmoid(out)
        # out = F.softmax(out, dim=1)
        return out


class Flowfront_CNN(nn.Module):
    def __init__(self):
        super(Flowfront_CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(1, 7, 7), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(8, 16, (1, 5, 5))
        self.conv3 = nn.Conv3d(16, 32, (1, 7, 7))
        self.conv4 = nn.Conv3d(32, 1, (1, 5, 5))
        self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.conv1(out)
        out = self.pool(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.drop(out)
        out = self.conv3(out)
        out = self.drop(out)
        out = self.conv4(out)
        return out


class FlowfrontToFiberfractionModel(nn.Module):
    def __init__(self):
        super(FlowfrontToFiberfractionModel, self).__init__()
        self.cnn = Flowfront_CNN()
        self.rnn = FlowfrontFeatures_RNN(input_dim=625)

    def forward(self, x):
        # print('>>> INFO: Forward pass CNN')
        out = self.cnn.forward(x)
        out = torch.squeeze(out, dim=1)
        out = out.permute(1, 0, 2, 3)
        out = out.reshape((out.size()[0], out.size()[1], -1)).contiguous()
        # print('>>> INFO: Forward pass RNN')
        out = self.rnn.forward(out)
        return out


if __name__ == "__main__":
    pass
    # if os.name == 'nt':
    #     data_root = Path(r'Y:\data\RTM\Lautern\output\with_shapes')
    # else:
    #     data_root = Path('/cfs/share/data/RTM/Lautern/output/with_shapes')

    # path = data_root / '2019-05-17_16-45-57_3000p'
    # paths = [path]
    # generator = pipeline.ERFH5_DataGenerator(
    #     paths, data_processing_function=dli.get_images_of_flow_front_and_permeability_map,
    #     data_gather_function=dg.get_filelist_within_folder,
    #     batch_size=1, epochs=1, max_queue_length=1, num_validation_samples=1)
    # model = FlowfrontToFiberfractionModel()
    # loss_criterion = nn.MSELoss()
    
    # for inputs, labels in generator:
    #     print("inputs", inputs.size())
    #     out = model(inputs)
    #     print(out)
    #     loss = loss_criterion(out, labels)
    #     print("loss", loss)
