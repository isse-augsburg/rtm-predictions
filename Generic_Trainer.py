import torch
import numpy as np 
import time
import erfh5_pipeline 
from collections import OrderedDict


class Master_Trainer():
    def __init__(self, model, generator:erfh5_pipeline.ERFH5_DataGenerator, loss_criterion = torch.nn.MSELoss(),train_print_frequency= 10, eval_frequency=100, savepath = "model.pth"):
        self.validationList = generator.get_validation_samples()
        self.model = model
        self.generator = generator
        self.train_print_frequency = train_print_frequency
        self.eval_frequency = eval_frequency
        self.savepath = savepath
        self.loss_criterion = loss_criterion
        self.loss_criterion = self.loss_criterion.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def start_training(self):
        self.__train()
        self.__eval()
        print(">>> INFO: MASTER PROCESS TERMINATED - TRAINING COMPLETE - MISSION SUCCESS ")


    def __train(self):
        start_time = time.time()
        for i, (inputs, labels) in enumerate(self.generator):
            #inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            #outputs = outputs.to(device, non_blocking=True)
            loss = self.loss_criterion(outputs, labels)
            #with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            self.optimizer.step()


            if i % self.train_print_frequency == 0:
                time_delta = time.time() - start_time
                print("Loss:", "{:12.4f}".format(loss.item()), "|| Duration of step:", "{:6}".format(
                    i), "{:10.2f}".format(time_delta), "seconds || Q:", self.generator.get_current_queue_length())
                start_time = time.time()

            if i % self.eval_frequency == 0:
                self.__eval()
                
    def __eval(self):
        with torch.no_grad():
                self.model.eval()
                loss = 0
                for sample in self.validationList:
                    data = sample[0].to(self.device)
                    label = sample[1].to(self.device)
                    label = torch.unsqueeze(label,0)
                    data = torch.unsqueeze(data, 0)
                    output = self.model(data)
                    loss = loss + self.loss_criterion(output, label).item()
                    #print(output.item(), label.item())

                loss = loss / len(self.validationList)
                print(">>> RMSE on Eval:", "{:8.4f}".format(np.sqrt(loss)))
                self.model.train()

    def saveModel(self):
        torch.save(self.model.state_dict(), self.savepath)

    
    def load_model(self):
        state_dict = torch.load(self.savepath)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params   
        self.model.load_state_dict(new_state_dict)
       

