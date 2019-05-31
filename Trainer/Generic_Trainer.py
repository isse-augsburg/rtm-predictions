import torch
import time
from Pipeline import erfh5_pipeline
from collections import OrderedDict
import numpy as np 
import math


class Master_Trainer():
    """Class that runs train and evaluation loops of PyTorch models automatically.

    Args: 
        model: PyTorch model that should be trained.
        generator: ERFH5_DataGenerator that provides the data.
        loss_criterion: Loss criterion for training.
        train_print_frequency: Frequency of printing the current loss, in iterations. 
        eval_frequenxy: Frequency of running a evaluation frequency on held out validation set, in iterations. 
        comment: Optional message that is printed to the command line, helps with understanding your experiments afterwards
        learning_rate: Optimizer's learning rate 
        classification_evaluator: Optional object for evaluating classification, see evaluation.py for more details 
    """
    def __init__(self, model, generator: erfh5_pipeline.ERFH5_DataGenerator, loss_criterion=torch.nn.MSELoss(),
                 train_print_frequency=10, eval_frequency=100, comment="No custom comment added.", 
                 learning_rate=0.00001, classification_evaluator=None):
        self.validationList = generator.get_validation_samples()
        self.model = model
        self.generator = generator
        self.train_print_frequency = train_print_frequency
        self.eval_frequency = eval_frequency
        self.loss_criterion = loss_criterion
        # self.loss_criterion = self.loss_criterion.cuda()
        self.learning_rate = learning_rate
        self.loss_criterion = loss_criterion.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.comment = comment
        self.classification_evaluator = classification_evaluator

    def start_training(self):
        """ Prints information about the used train config and starts the training of the trainer's model
        """
        self.__print_info()
        self.__print_comment()
        self.__train()
        self.__eval()
        #print(">>> INFO: MASTER PROCESS TERMINATED - TRAINING COMPLETE - MISSION SUCCESS ")
        print(">>> INFO: TRAINING COMPLETE.")
    
    def __print_info(self): 
        print("###########################################")
        print(">>> Model Trainer INFO <<<")
        print("Loss criterion:", self.loss_criterion)
        print("Optimizer:", self.optimizer)
        print("Learning rate:", self.learning_rate)
        print("Evaluation frequency:", self.eval_frequency)
        print("Model:", self.model)
        print("Model savepath (may not be used):", self.savepath)
        print("Evaluation function (optional):", self.eval_func)
        print("###########################################")


    def __print_comment(self):
        print("###########################################")
        print(self.comment)
        print("###########################################")

    def __train(self):
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(self.generator):
            # inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # outputs = outputs.to(device, non_blocking=True)
            loss = self.loss_criterion(outputs, labels)
            # with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
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
            for i, sample in enumerate(self.validationList):
                data = sample[0].to(self.device)
                label = sample[1].to(self.device)
                label = torch.unsqueeze(label, 0)
                data = torch.unsqueeze(data, 0)
                output = self.model(data)
                #print(output,label)
                l = self.loss_criterion(output, label).item()
                loss = loss + l
                
                if self.classification_evaluator is not None: 
                    self.classification_evaluator.commit(output.cpu(), label.cpu())

                # print("loss:", l)
                # print(output.item(), label.item())

            loss = loss / len(self.validationList)
            print(">>> Mean Loss on Eval:", "{:8.4f}".format(loss))
            
            if self.classification_evaluator is not None:
                self.classification_evaluator.print_metrics() 
                self.classification_evaluator.reset()
        
            self.model.train()

    

    def save_model(self, savepath):
        """Saves the model. 

        Args: 
            savepath (string): Path and filename to the model, eg 'model.pt' 
        """
        torch.save(self.model.state_dict(), savepath)


    def load_model(self, modelpath):
        """Loads the parameters of a previously saved model. See the official PyTorch docs for more details.

        ARgs: 
            modelpath (string): Path to the stored model.
        """
        state_dict = torch.load(modelpath, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params   
        self.model.load_state_dict(new_state_dict)
