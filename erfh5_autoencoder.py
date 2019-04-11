import torch
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F
import erfh5_pipeline as pipeline
import time
import numpy as np

from collections import OrderedDict
import matplotlib.pyplot as plt 
import matplotlib

#from apex import amp

#amp_handle = amp.init() 


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
            x = F.relu(layer(x))
        return x


class erfh5_Distributed_Autoencoder(nn.Module):
    def __init__(self, dgx_mode=True, layers_size_list=[69366, 15000, 8192]):
        super(erfh5_Distributed_Autoencoder, self).__init__()

        self.encoder = stacked_FullyConnected(layers_size_list)
        self.decoder = stacked_FullyConnected(reversed(layers_size_list))
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
        return x

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
    batch_size = 8
    epochs = 1
    eval_frequency = 10
    test_frequency = 50
    path = '/cfs/home/l/o/lodesluk/models/encoder-08-04-1304.pth'
    #path =  '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/models/encoder-08-04-1304.pth'
  
    try:
        generator = pipeline.ERFH5_DataGenerator(data_path='/cfs/share/data/RTM/Lautern/clean_erfh5/', batch_size=batch_size,
                                                    epochs=epochs, indices=1, max_queue_length=32, pipeline_mode=pipeline.Pipeline_Mode.single_instance, num_validation_samples=3)
        validation_samples = generator.get_validation_samples()
    except Exception as e:
        print("Fatal Error:", e)
        exit() 

    #encoder = erfh5_Autoencoder(69366, [8000, 6000])
    
    #autoencoder = erfh5_Distributed_Autoencoder()
    half_encoder = load_stacked_fc(path)
    
    loss_criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.000005)
    
    

    #encoder, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    """ start_time = time.time()
    counter = 1

    print("Expected length of data generator:", len(generator))

    for inputs, _ in generator:

        inputs = torch.FloatTensor(inputs)

        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        outputs = outputs.to(device)

        loss = loss_criterion(outputs, inputs)

        with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if counter % eval_frequency == 0:
            time_per_epoch = time.time() - start_time
            print("Loss:", "{:8.4f}".format(loss.item()), "|| Duration of step:", "{:6}".format(
                counter), "{:4.2f}".format(time_per_epoch), "seconds || Q:", generator.get_current_queue_length())
            start_time = time.time()

        if counter % test_frequency == 0:
            with torch.no_grad():
                autoencoder.eval()
                loss = 0
                for i in validation_samples:
                    i = torch.FloatTensor(i)
                    i = i.to(device)
                    i = torch.unsqueeze(i, 0)
                    output = autoencoder(i)
                    output = output.to(device) 
                    loss = loss + loss_criterion(output, i).item()

                loss = loss / len(validation_samples)
                print(">>>Loss on Test:", "{:8.4f}".format(loss))
                autoencoder.train()

        counter = counter + 1


    with torch.no_grad():
        autoencoder.eval()
        loss = 0
        for i in validation_samples:
            i = torch.FloatTensor(i)
            i = i.to(device)
            i = torch.unsqueeze(i, 0)
            output = autoencoder(i)
            output = output.to(device) 
            loss = loss + loss_criterion(output, i).item()

        loss = loss / len(validation_samples)
        print(">>>FINAL on Test:", "{:8.4f}".format(loss))
        autoencoder.train()

    print(">>>INFO: Saving state dict")
    #torch.save(encoder.state_dict(), path)
    autoencoder.save_encoder(path) """




    print(">>>INFO: Loading State dict finished.")  
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


    print(">>>INFO: Finished.")