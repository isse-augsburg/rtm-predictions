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
                 train_print_frequency=10, eval_frequency=100, savepath="model.pth", eval_func=None,
                 comment="No custom comment added.", learning_rate=0.00001, calc_metrics=False, classification_evaluator=None,
                 imsize=(155, 155)):
        self.validationList = generator.get_validation_samples()
        self.model = model
        self.generator = generator
        self.train_print_frequency = train_print_frequency
        self.eval_frequency = eval_frequency
        self.savepath = savepath
        self.loss_criterion = loss_criterion
        self.learning_rate = learning_rate
        self.loss_criterion = loss_criterion.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.comment = comment
        self.eval_func = eval_func
        self.calc_metrics = calc_metrics
        self.imsize = imsize
        self.classification_evaluator = classification_evaluator

    def start_training(self):
        """ Prints information about the used train config and starts the training of the trainer's model
        """
        self.__print_info()
        # self.__print_comment()
        self.__train()
        print('Test set missing. So no testing.')
        # self.__eval()
        print(">>> INFO: TRAINING COMPLETE.")
    
    def __print_info(self): 
        print("###########################################")
        print(">>> Model Trainer INFO <<<")
        print("Loss criterion:", self.loss_criterion)
        print("Optimizer:", self.optimizer)
        print("Learning rate:", self.learning_rate)
        print("Evaluation frequency:", self.eval_frequency)
        print("Model:", self.model)
        print("###########################################")

    def __print_comment(self):
        print("###########################################")
        print(self.comment)
        print("###########################################")

    def __train(self):
        start_time = time.time()
        eval_step = 0
        time_sum = 0
        i_of_epoch = 0
        for i, (inputs, label) in enumerate(self.generator):
            i_of_epoch += 1
            inputs = inputs.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            label = label.reshape(label.shape[0], self.imsize[0] * self.imsize[1])
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # print(inputs.shape)
            label = torch.stack([label] * len(outputs))
            label = label / 255
            loss = self.loss_criterion(outputs, label)
            loss.backward()
            self.optimizer.step()
            if i % self.train_print_frequency == 0 and i != 0:
                time_delta = time.time() - start_time
                time_sum += time_delta
                # print(time_sum, i)
                print(f"Loss: {loss.item():12.4f} || Duration of step {i:6}: {time_delta:10.2f} seconds; avg: {time_sum / i_of_epoch:10.2f}|| Q: {self.generator.get_current_queue_length()}")
                start_time = time.time()

            if i % self.eval_frequency == 0 and i != 0:
                self.__eval(eval_step)
                time_sum = 0
                eval_step += 1
                i_of_epoch = 0

    def __eval(self, eval_step=0):
        # print('EVAL')
        with torch.no_grad():
            self.model.eval()
            tp, fp, tn, fn = 0, 0, 0, 0
            loss = 0
            for i, sample in enumerate(self.validationList):
                data = sample[0].to(self.device)
                label = sample[1].to(self.device)
                label = label.reshape((self.imsize[0] * self.imsize[1]))
                label = torch.stack([label] * len(data))
                # label = torch.sigmoid(label) Loses a lot of information
                label = label / 255
                data = torch.unsqueeze(data, 0)
                output = self.model(data)
# <<<<<<< Trainer/Generic_Trainer.py
                loss = self.loss_criterion(output, label).item()
                print(f"Sample {i}: validation --- Loss: {loss:12.4f}")

                if self.eval_func is not None:
                    self.eval_func(output.cpu(), label.cpu(), f'{eval_step}_{i}')
                
                if not self.calc_metrics:
                    continue
                prediction = np.around(output.cpu())
                # confusion_matrix[int(label[0][0].cpu())][int(prediction[0][0].cpu())] +=1

                if np.array_equal(prediction.cpu(), label.cpu()):
                    if prediction[0][0] == 1:
                        tp += 1
                    else: tn += 1
                else:
                    if prediction[0][0] == 1:
                        fp += 1
                    else: fn += 1

                # print("loss:", l)
                # print(output.item(), label.item())

            # loss = loss / len(self.validationList)
            # print(">>> Mean Loss on Eval:", "{:8.4f}".format(loss))
            if self.calc_metrics:
                print(">>>True positives:", tp, ">False positives:", fp, ">True negatives:", tn, ">False negatives:", fn)
                print(">>>Accuracy:", self.__calc_accuracy(tp, fp, tn, fn), ">Precision:", self.__calc_precision(tp, fp, tn, fn), ">Recall:", self.__calc_recall(tp, fp, tn, fn))
                # print(">>>Confusion matrix:", confusion_matrix)
# =======
#                 #print(output,label)
#                 l = self.loss_criterion(output, label).item()
#                 loss = loss + l
                
#                 if self.classification_evaluator is not None: 
#                     self.classification_evaluator.commit(output.cpu(), label.cpu())

#                 # print("loss:", l)
#                 # print(output.item(), label.item())

#             loss = loss / len(self.validationList)
#             print(">>> Mean Loss on Eval:", "{:8.4f}".format(loss))
            
#             if self.classification_evaluator is not None:
#                 self.classification_evaluator.print_metrics() 
#                 self.classification_evaluator.reset()
        
# >>>>>>> Trainer/Generic_Trainer.py
            self.model.train()

    

    def save_model(self, savepath):
        """Saves the model. 

        Args: 
            savepath (string): Path and filename to the model, eg 'model.pt' 
        """
        torch.save(self.model.state_dict(), savepath)



    def __calc_recall(self, tp, fp, tn, fn): 
        return (tp) / max((tp + fn), 0.00000001)


    def load_model(self, modelpath):
        """Loads the parameters of a previously saved model. See the official PyTorch docs for more details.
>>>>>>> Trainer/Generic_Trainer.py

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
