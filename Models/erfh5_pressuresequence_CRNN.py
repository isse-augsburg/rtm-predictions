import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class ERFH5_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, batch_size=8, num_layers=4):
        super(ERFH5_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nlayers = num_layers

        self.lstm = nn.LSTM(input_dim, self.hidden_dim,
                            batch_first=False,
                            num_layers=self.nlayers,
                            bidirectional=False,
                            dropout=0.0
                            )

        # Fully connected for decision making after LSTM
        self.hidden2hidden1 = nn.Linear(int(hidden_dim), 750)
        self.hidden2hidden2 = nn.Linear(750, 600)
        self.hidden2hidden3 = nn.Linear(600, 300)
        self.hidden2hidden4 = nn.Linear(300, 150)
        self.hidden2hidden5 = nn.Linear(150, 50)
        self.hidden2value = nn.Linear(50, 1)
        self.drop = nn.Dropout(0.0)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.hidden2hidden1.bias.data.fill_(0)
        self.hidden2hidden1.weight.data.uniform_(-initrange, initrange)
        self.hidden2hidden2.bias.data.fill_(0)
        self.hidden2hidden2.weight.data.uniform_(-initrange, initrange)
        self.hidden2hidden3.bias.data.fill_(0)
        self.hidden2hidden3.weight.data.uniform_(-initrange, initrange)
        self.hidden2hidden4.bias.data.fill_(0)
        self.hidden2hidden4.weight.data.uniform_(-initrange, initrange)
        self.hidden2hidden5.bias.data.fill_(0)
        self.hidden2hidden5.weight.data.uniform_(-initrange, initrange)
        self.hidden2value.bias.data.fill_(0)
        self.hidden2value.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        return [
            Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_dim)),
            Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_dim)),
        ]

    def forward(self, x):
        # for i in range(len(hidden)):
            # hidden[i] = hidden[i].permute(1, 0, 2).contiguous() """

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
        out = F.relu(self.hidden2hidden4(out))
        out = self.drop(out)
        out = F.relu(self.hidden2hidden5(out))
        out = self.hidden2value(out)
        # out = F.softmax(out)
        out = torch.sigmoid(out)
        return out


class ERFH5_Pressure_CNN(nn.Module):
    def __init__(self, input_channels=50):
        super(ERFH5_Pressure_CNN, self).__init__()
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(self.input_channels, 64, (5, 5), dilation=1)
        self.conv2 = nn.Conv2d(64, 128, (3, 3))
        self.conv3 = nn.Conv2d(128, 256, (5, 5))

        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.pool(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.drop(out)
        out = self.conv3(out)
        out = self.drop(out)
        return out


class ERFH5_Pressure_FC(nn.Module):

    def __init__(self, input_features=24576):
        super(ERFH5_Pressure_FC, self).__init__()
        self.input_features = input_features

        self.in_layer = nn.Linear(self.input_features, 8192)
        self.hidden1 = nn.Linear(8192, 4096)
        self.hidden2 = nn.Linear(4096, 2048)
        self.hidden3 = nn.Linear(2048, 1024)
        self.hidden4 = nn.Linear(1024, 512)
        self.hidden5 = nn.Linear(512, 256)
        self.hidden6 = nn.Linear(256, 128)
        self.out_layer = nn.Linear(128, 1)

    def init_weights(self):
        init_range = 0.1

        self.in_layer.bias.data.fill_(0)
        self.in_layer.weight.data.uniform_(-init_range, init_range)
        self.hidden1.bias.data.fill_(0)
        self.hidden1.weight.data.uniform_(-init_range, init_range)
        self.hidden2.bias.data.fill_(0)
        self.hidden2.weight.data.uniform_(-init_range, init_range)
        self.hidden3.bias.data.fill_(0)
        self.hidden3.weight.data.uniform_(-init_range, init_range)
        self.hidden4.bias.data.fill_(0)
        self.hidden4.weight.data.uniform(-init_range, init_range)
        self.hidden5.bias.data.fill_(0)
        self.hidden5.weight.data.uniform_(-init_range, init_range)
        self.hidden6.bias.data.fill_(0)
        self.hidden6.weight.data.uniform(-init_range, init_range)
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform(-init_range, init_range)

    def forward(self, x):
        # TODO dropout!

        out = F.relu(self.in_layer(x))
        out = F.relu(self.hidden1(out))
        out = F.relu(self.hidden2(out))
        out = F.relu(self.hidden3(out))
        out = F.relu(self.hidden4(out))
        out = F.relu(self.hidden5(out))
        out = F.relu(self.hidden6(out))
        out = torch.sigmoid(self.out_layer(out))
        return out


class ERFH5_PressureSequence_Model(nn.Module):
    def __init__(self):
        super(ERFH5_PressureSequence_Model, self).__init__()
        self.cnn = ERFH5_Pressure_CNN()
        # self.rnn = ERFH5_RNN(input_dim=96)
        self.fc = ERFH5_Pressure_FC()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.cnn(x)

        # out = out.reshape((out.size()[0], out.size()[1], -1))
        out = out.reshape((out.size()[0], -1))
        # out = self.rnn(out)

        out = self.fc(out)

        return out


if __name__ == "__main__":
    path = ['/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p']
    generator = pipeline.ERFH5_DataGenerator(
        path,
        data_processing_function=dl.get_sensordOata_and_filling_percentage,
        data_gather_function=dg.get_filelist_within_folder,
        batch_size=1, epochs=1, max_queue_length=32, num_validation_samples=1)
    model = ERFH5_PressureSequence_Model()
    loss_criterion = torch.nn.MSELoss()

    for inputs, labels in generator:
        print("inputs", inputs.size())
        out = model(inputs)
        print(out)
        loss = loss_criterion(out, labels)
        print("loss", loss)
