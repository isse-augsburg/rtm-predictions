import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
import erfh5_pipeline as pipeline
import time

import numpy as np

batchsize = 16
epochs = 80
eval_frequency = 2





class Net(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, num_layers=2):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nlayers = num_layers
       
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=False, num_layers=self.nlayers, bidirectional=True)

        self.hidden2hidden1 = nn.Linear(int(hidden_dim*2), 1024)
        self.hidden2hidden2 = nn.Linear(1024, 512)
        self.hidden2hidden3 = nn.Linear(512, 256)
        self.hidden2hidden4 = nn.Linear(256, 128)
        self.hidden2value = nn.Linear(128, 1)

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
            Variable(torch.zeros( self.nlayers, self.batch_size, self.hidden_dim)),
            Variable(torch.zeros( self.nlayers, self.batch_size, self.hidden_dim)),
        ]

    def forward(self, x):
        """  for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous() """

        x = x.permute(1, 0, 2)
        lstm_out, hidden = self.lstm(x)
       
        out = lstm_out[-1]
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.hidden2hidden1(out))
        out = F.relu(self.hidden2hidden2(out))
        out = F.relu(self.hidden2hidden3(out))
        out = F.relu(self.hidden2hidden4(out))
        out = self.hidden2value(out)
        return out


try:
    generator = pipeline.ERFH5_DataGenerator(
       '/cfs/share/data/RTM/Lautern/clean_erfh5/', batch_size=batchsize, epochs=epochs,indices=range(100) ,max_queue_length=128)
except Exception as e:
    print("Fatal Error:", e)
    exit()

# '/home/lodes/Sim_Results'
# '/cfs/share/data/RTM/Lautern/clean_erfh5/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Net(69366, 1024, batchsize)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)


model.to(device)


loss_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

start_time = time.time()
counter = 1

print("Expected length of data generator:", len(generator))


for inputs, labels in generator:


    inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    
    loss = loss_criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()

    if counter % eval_frequency == 0:
        time_per_epoch = time.time() - start_time
        print("Loss:", "{:12.4f}".format(loss.item()), "|| Duration of step:", "{:6}".format(
            counter), "{:10.2f}".format(time_per_epoch), "seconds || Q:", generator.get_current_Q_length())
        start_time = time.time()

    counter = counter + 1

print(counter)
