import torch 
import torch.nn as nn 
import torch.nn.functional as F
import erfh5_pipeline as pipeline
import time  
import numpy as np 

batchsize = 4
epochs = 5
eval_frequency = 2

class Net(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size):
        super(Net, self).__init__()     
        self.hidden_dim = hidden_dim 
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2value = nn.Linear(hidden_dim, 1)
   

    def forward(self, x):
        # Max pooling over a (2, 2) window
        lstm_out, hidden = self.lstm(x)
        value = self.hidden2value(lstm_out[-1].view(self.batch_size, -1))
        return value 


generator = pipeline.ERFH5_DataGenerator('/home/lodes/Sim_Results', batch_size=batchsize, epochs=epochs)



model = Net(69366, 10, 4)
loss_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

start_time = time.time()
counter = 1

print("Expected length of data generator:", len(generator))

for inputs, labels in generator:
    
    

    indices = [0, 20, 40, 60, 80]
    inputs = [[i[j] for j in indices] for i in inputs]
    inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
    inputs = inputs.permute(1, 0, 2)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_criterion(outputs, labels)
    loss.backward() 
    optimizer.step()
    
    if counter % eval_frequency == 0:
        time_per_epoch = time.time() - start_time
        print("Loss:" , "{:12.4f}".format(loss.item()), "|| Duration of step:" , "{:6}".format(counter), "{:10.2f}".format(time_per_epoch), "seconds")
        start_time = time.time()
    
    
    counter = counter + 1
    
print(counter)