import torch
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F
import erfh5_pipeline as pipeline
import time
import numpy as np

import data_loaders as dl

from collections import OrderedDict
import matplotlib.pyplot as plt 
import matplotlib




class stacked_FullyConnected(nn.Module):
    def __init__(self, FC_List=[500, 200, 100]):
        super(stacked_FullyConnected, self).__init__()
        self.FC_List = FC_List
        self.FCs = nn.ModuleList()
        self.__get_fc()

    def __get_fc(self):
        s = self.FC_List[0]
        num = self.FC_List[1]
        self.FCs.append(nn.Linear(s, num))
        s = num
        for num in self.FC_List[2:]:
            self.FCs.append(nn.Dropout(p=0.5))
            self.FCs.append(nn.Linear(s, num))
            s = num

    def forward(self, inputs):
        x = inputs
        for layer in self.FCs:
            x = F.softsign(layer(x))
        return x


class erfh5_Distributed_Autoencoder(nn.Module):
    def __init__(self, dgx_mode=True, layers_size_list=[69366, 15000]):
        super(erfh5_Distributed_Autoencoder, self).__init__()

        self.encoder = stacked_FullyConnected(layers_size_list)
        self.decoder = stacked_FullyConnected(list(reversed(layers_size_list)))
        print(self.encoder)
        print(self.decoder)

        if dgx_mode:
            self.encoder = nn.DataParallel(self.encoder, device_ids=[
                                           0, 1, 2, 3]).to('cuda:0')
            self.decoder = nn.DataParallel(self.decoder, device_ids=[
                                           4, 5, 6, 7]).to('cuda:4')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.to('cuda:0')

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)


class erfh5_Autoencoder(nn.Module):
    def __init__(self, input_size, FC_List=[500, 200, 100]):
        super(erfh5_Autoencoder, self).__init__()
        self.FC_List = FC_List
        self.input_size = input_size
        self.FCs = nn.ModuleList()

        self.__get_fc()

        #self.weightList = nn.ParameterList([nn.Parameter(f.weight) for f in self.FCs])
        #self.biasList = nn.ParameterList([nn.Parameter(f.bias) for f in self.FCs])
        [print(f) for f in self.FCs]

    def __get_fc(self):
        s = self.input_size
        for num in self.FC_List:
            self.FCs.append(nn.Linear(s, num))
            self.FCs.append(nn.Dropout(p=0.5))
            s = num
        for num in reversed(self.FC_List[:-1]):
            self.FCs.append(nn.Linear(s, num))
            self.FCs.append(nn.Dropout(p=0.5))
            s = num
        self.FCs.append(nn.Linear(s, self.input_size))

    def forward(self, inputs):
        x = inputs
        for layer in self.FCs:
            x = F.relu(layer(x))
        return x

    def get_encoding(self):
        return self.FCs[int((self.FCs.__len__()-1)/2)]

# '/home/lodes/Sim_Results'
# '/cfs/share/data/RTM/Lautern/clean_erfh5/'


def load_stacked_fc(path, list=[69366, 15000, 8192]):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    model = stacked_FullyConnected(list)

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params   
    model.load_state_dict(new_state_dict)
    return model 




if __name__ == "__main__":
    pass 

   # half_encoder = load_stacked_fc(path)


    """ print(">>>INFO: Loading State dict finished.")  
    half_encoder.to(device)  
    with torch.no_grad():
        half_encoder.eval()
        loss = 0
        counter = 0
        for i in validation_samples:
            i = torch.FloatTensor(i)
            i = i.to(device)
            i = torch.unsqueeze(i, 0)
            output = half_encoder(i)
            #output = output.to(device) 
            #loss = loss + loss_criterion(output, i).item()
            output = output.cpu().numpy()
            i = i.cpu().numpy()
            
            plt.figure()
            plt.subplot(211)
            plt.plot(i, 'bo')
            
            plt.subplot(212)
            plt.plot(output, 'ro')
            plt.savefig('/cfs/home/l/o/lodesluk/models/' + str(counter) + '.png')
            print("plot saved")
            counter = counter + 1
            

        #loss = loss / len(validation_samples)
        #print(">>>Loss on loaded model:", "{:8.4f}".format(loss))
        half_encoder.train()
 """

    print(">>>INFO: Finished.")