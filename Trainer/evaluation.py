import numpy as np 
from PIL import Image
import math 

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