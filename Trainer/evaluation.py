import os
from functools import partial
from multiprocessing.pool import Pool

from PIL import Image
import math 
from pathlib import Path
import numpy as np


def pixel_wise_loss_multi_input_single_label(input, target):
    print('Loss')
    loss = 0
    for el in input:
        out = el - target
        # out = out * weights.expand_as(out)
        loss += out.sum(0)
    return loss


def plot_predictions_and_label(input, target, _str):
    if os.name == 'nt':
        debug_path = Path(r'X:\s\t\stiebesi\code\debug\overfit')
    else:
        debug_path = Path('/cfs/home/s/t/stiebesi/code/debug/overfit/')
    (debug_path / 'predict').mkdir(parents=True, exist_ok=True)

    x = input.reshape(input.shape[0], 155, 155)
    x = x * 255
    with Pool() as p:
        p.map(partial(save_img, debug_path / 'predict', _str, x), range(0, input.shape[0], 1))
    y = target.reshape(target.shape[0], 155, 155)
    y = y * 255
    im = Image.fromarray(np.asarray(y[0]))
    path = debug_path / 'label'
    path.mkdir(parents=True, exist_ok=True)
    file = f'{_str}.png'
    im.convert('RGB').save(path / file)
    im.close()


def save_img(path, _str, x, index):
    try:
        im = Image.fromarray(np.asarray(x[index]))
        file = f'{_str}_{index}.png'
        im.convert('RGB').save(path / file)
        im.close()
    except KeyError:
        print('ERROR: save_img')



class Binary_Classification_Evaluator(): 
    """Evaluator specifically for binary classification. Calculates common metrices and a confusion matrix.
    """

    def __init__(self): 
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0 
        self.confusion_matrix = np.zeros((2, 2), dtype=int)

    def commit(self, net_output, label):
        """Updates the confusion matrix and updates the metrics. 

        Args: 
            net_output: single prediction of the model. 
            label: single label for the prediction.
        """

        if math.isnan(net_output[0][0]):
            return 

        prediction = np.around(net_output)

        self.confusion_matrix[int(label[0][0].cpu())][int(prediction[0][0].cpu())] += 1

        if np.array_equal(prediction, label):
            if prediction[0][0] == 1: 
                self.tp += 1 
            else: self.tn += 1
        else: 
            if prediction[0][0] == 1: 
                self.fp += 1
            else: self.fn += 1

    def print_metrics(self): 
        """Prints the counts of True/False Positives and True/False Negatives, Accuracy, Precision, Recall, Specificity and the confusion matrix.
        """

        print(">>>True positives:", self.tp, ">False positives:", self.fp, ">True negatives:", self.tn, ">False negatives:", self.fn)
        print(">>>Accuracy:", "{:7.4f}".format(self.__calc_accuracy(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)), 
            ">Precision:", "{:7.4f}".format(self.__calc_precision(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)), 
            ">Recall:", "{:7.4f}".format(self.__calc_recall(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)), 
            ">Specificity:", "{:7.4f}".format(self.__calc_specificity(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)))
        print(">>>Confusion matrix:", self.confusion_matrix)

    def reset(self): 
        """Resets the internal counters for the next evaluation loop. 
        """ 
        
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0 
        self.confusion_matrix = np.zeros((2, 2), dtype=int)

    def __calc_accuracy(self, tp, fp, tn, fn): 
        return (tp + tn) / max((tp + tn + fp + fn), 0.00000001) 

    def __calc_precision(self, tp, fp, tn, fn): 
        return (tp) / max((tp + fp), 0.00000001)

    def __calc_recall(self, tp, fp, tn,fn): 
        return (tp) / max((tp + fn), 0.00000001)

    def __calc_specificity(self, tp, fp, tn, fn): 
        return (tn) / max((tn + fp), 0.00000001)


class FlowFront_Prediction_Evaluator(): 
    def __init__(self, name, path="/cfs/home/s/c/schroeni/Data/Eval/"):
        self.name = name 
        self.path = path 

    def commit(self, inputs, label):
        inputs = np.squeeze(inputs)
        label = np.squeeze(label)
        inp = Image.fromarray(np.uint8((inputs) * 255))
        lab = Image.fromarray(np.uint8((label) * 255))
        inp.save(self.path + "inp_" + str(self.name) + ".bmp")
        lab.save(self.path + "lab_" + str(self.name) + ".bmp")

    def print_metrics(self):
        pass 

    def reset(self):
        pass 
