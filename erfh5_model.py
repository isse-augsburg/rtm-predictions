import torch 
import torch.nn as nn 
import torch.nn.functional as F
import erfh5_pipeline as pipeline
import time  
import numpy as np 

batchsize = 8
epochs = 1
steps_per_epoch = 500

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        """ self.conv1 = nn.Conv1d(in_channels=1, kernel_size=5, out_channels=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5,)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5,) """
        self.fc1 = nn.Linear(69366, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        """ x = self.pool(F.relu(self.conv1(x)))        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x)) """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


generator = pipeline.ERFH5_DataGenerator('/home/lodes/Sim_Results', batch_size=batchsize)



model = Net()
loss_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

for e in range(epochs):
    start_time = time.time()
    counter = 1
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            #inputs, labels = inputs.to(device), labels.to(device)
    
            inputs, labels = generator.get_batch()
            inputs = [i[90] for i in inputs]
            inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
            train_acc = 0 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            counter = counter + 1
            print("Loss: " + str(loss.item()))

        time_per_epoch = time.time() - start_time
        print("Duration of epoch " + str(e+1) + ": " + str(time_per_epoch))