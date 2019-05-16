import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Pipeline import erfh5_pipeline as pipeline, data_loaders as dl


class ERFH5_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, batch_size=8, num_layers=3):
        super(ERFH5_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nlayers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=False, num_layers=self.nlayers, bidirectional=False, dropout=0.2)

        self.hidden2hidden1 = nn.Linear(int(hidden_dim), 600)
        self.hidden2hidden2 = nn.Linear(600, 400)
        self.hidden2hidden3 = nn.Linear(400, 200)
        self.hidden2hidden4 = nn.Linear(200, 100)
        self.hidden2value = nn.Linear(100, 2)
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
        self.hidden2hidden4.bias.data.fill_(0)
        self.hidden2hidden4.weight.data.uniform_(-initrange, initrange)
        self.hidden2value.bias.data.fill_(0)
        self.hidden2value.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        return [
            Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_dim)),
            Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_dim)),
        ]

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
        out = F.relu(self.hidden2hidden4(out))
        out = self.drop(out)
        out = self.hidden2value(out)
        out = F.softmax(out)
        return out


class ERFH5_Pressure_CNN(nn.Module):
    def __init__(self):
        super(ERFH5_Pressure_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 7), dilation=1)
        self.conv2 = nn.Conv2d(8, 16, (1, 5))
        self.conv3 = nn.Conv2d(16, 32, (1, 7))
        self.conv4 = nn.Conv2d(32, 64, (1, 5))
        self.pool = nn.MaxPool2d((2, 2), (2, 2))
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
        return out


class ERFH5_PressureSequence_Model(nn.Module):
    def __init__(self):
        super(ERFH5_PressureSequence_Model, self).__init__()
        self.cnn = ERFH5_Pressure_CNN()
        self.rnn = ERFH5_RNN(input_dim=2560)

    def forward(self, x):
        out = self.cnn(x)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape((out.size()[0], out.size()[1], -1))
        out = self.rnn(out)
        return out


if __name__ == "__main__":
    path = [
        '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-04-23_13-00-58_200p/']
    generator = pipeline.ERFH5_DataGenerator(
        path, data_processing_function=dl.get_sensordata_and_filling_percentage,
        data_gather_function=dl.get_filelist_within_folder,
        batch_size=1, epochs=1, max_queue_length=32, num_validation_samples=1)
    model = ERFH5_PressureSequence_Model()
    loss_criterion = torch.nn.MSELoss()

    for inputs, labels in generator:
        print("inputs", inputs.size())
        out = model(inputs)
        print(out)
        loss = loss_criterion(out, labels)
        print("loss", loss)
