import logging
import math
import os
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

""" 
>>>> PLEASE NOTE: <<<<
Evaluation classes must provide three functions even if not all of them have functionality: 

* commit(output, label): updates the evaluation class with a new pair of a single prediction and a single label
* print_metrics(): prints a set of application-specific print_metrics
* reset: Resets the internal metrics of an evaluator, e.g. after a evaluation loop is finished.  

They have to be given a save_path, where the results are going to be stored.
"""


class Evaluator:
    def __init__(self):
        pass

    def commit(self, *args, **kwargs):
        pass

    def print_metrics(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


def pixel_wise_loss_multi_input_single_label(input, target):
    loss = 0
    for el in input:
        out = el - target
        # out = out * weights.expand_as(out)
        loss += out.sum(0)
    return loss


def plot_predictions_and_label(input, target, _str):
    if os.name == "nt":
        debug_path = Path(r"X:\s\t\stiebesi\code\debug\overfit")
    else:
        debug_path = Path("/cfs/home/s/t/stiebesi/code/debug/overfit/")
    (debug_path / "predict").mkdir(parents=True, exist_ok=True)

    x = input.reshape(input.shape[0], 155, 155)
    x = x * 255
    with Pool() as p:
        p.map(
            partial(save_img, debug_path / "predict", _str, x),
            range(0, input.shape[0], 1),
        )
    y = target.reshape(target.shape[0], 155, 155)
    y = y * 255
    im = Image.fromarray(np.asarray(y[0]))
    path = debug_path / "label"
    path.mkdir(parents=True, exist_ok=True)
    file = f"{_str}.png"
    im.convert("RGB").save(path / file)
    im.close()


def save_img(path, _str, x, index):
    try:
        im = Image.fromarray(np.asarray(x[index]))
        file = f"{_str}_{index}.png"
        im.convert("RGB").save(path / file)
        im.close()
    except KeyError:
        logger = logging.getLogger(__name__)
        logger.error("ERROR: save_img")


class SensorToFlowfrontEvaluator(Evaluator):
    def __init__(self,
                 save_path=Path("/home/schroeter/Desktop/output"),
                 halving_factor=0,
                 skip_images=False):
        self.num = 0
        self.save_path = save_path
        self.im_save_path = save_path / "images"
        self.im_save_path.mkdir(parents=True, exist_ok=True)
        self.halving_factor = halving_factor
        self.skip_images = skip_images

    def commit(self, net_output, label, inputs, *args):
        if self.skip_images:
            return
        a = net_output.numpy()
        a = np.squeeze(a)
        b = label.numpy()
        b = np.squeeze(b)
        c = inputs.numpy()
        c = np.squeeze(c)
        c = c.reshape(38, 30)
        if self.halving_factor != 0:
            c = c[::self.halving_factor, ::self.halving_factor]

        plt.imsave(self.im_save_path / Path(str(self.num) + "out.jpg"), a)
        plt.imsave(self.im_save_path / Path(str(self.num) + "lab.jpg"), b)
        plt.imsave(self.im_save_path / Path(str(self.num) + "inp.jpg"), c)

        self.num += 1
        pass

    def print_metrics(self):
        pass

    def reset(self):
        self.num = 0
        pass


class BinaryClassificationEvaluator(Evaluator):
    """Evaluator specifically for binary classification. Calculates common metrices and a confusion matrix.
    """

    def __init__(self, save_path=None):
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)
        self.save_path = save_path

    def commit(self, net_output, label, *args):
        """Updates the confusion matrix and updates the metrics. 

        Args: 
            net_output: single prediction of the model. 
            label: single label for the prediction.
        """

        if math.isnan(net_output[0]):
            return

        prediction = np.around(net_output)

        self.confusion_matrix[int(label[0].cpu())][
            int(prediction[0].cpu())] += 1

        if np.array_equal(prediction, label):
            if prediction[0] == 1:
                self.tp += 1
            else:
                self.tn += 1
        else:
            if prediction[0] == 1:
                self.fp += 1
            else:
                self.fn += 1

    def print_metrics(self):
        """Prints the counts of True/False Positives and True/False Negatives, Accuracy, Precision, Recall,
        Specificity and the confusion matrix.
        """
        logger = logging.getLogger(__name__)

        logger.info(
            "True positives: %s, False positives: %s, True negatives: %s, False negatives: %s",
            str(self.tp),
            str(self.fp),
            str(self.tn),
            str(self.fn)
        )

        logger.info(
            "Accuracy: %s, Precision: %s, Recall: %s, Specificity: %s",
            "{:7.4f}".format(
                self.__calc_accuracy(tp=self.tp, fp=self.fp, tn=self.tn,
                                     fn=self.fn)
            ),
            "{:7.4f}".format(
                self.__calc_precision(tp=self.tp, fp=self.fp, tn=self.tn,
                                      fn=self.fn)
            ),
            "{:7.4f}".format(
                self.__calc_recall(tp=self.tp, fp=self.fp, tn=self.tn,
                                   fn=self.fn)
            ),
            "{:7.4f}".format(
                self.__calc_specificity(tp=self.tp, fp=self.fp, tn=self.tn,
                                        fn=self.fn)
            ),
        )

        logger.info("Confusion matrix:\n%s", str(self.confusion_matrix))

    def reset(self):
        """Resets the internal counters for the next evaluation loop. 
        """

        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)

    @staticmethod
    def __calc_accuracy(tp, fp, tn, fn):
        return (tp + tn) / max((tp + tn + fp + fn), 0.00000001)

    @staticmethod
    def __calc_precision(tp, fp, tn, fn):
        return tp / max((tp + fp), 0.00000001)

    @staticmethod
    def __calc_recall(tp, fp, tn, fn):
        return tp / max((tp + fn), 0.00000001)

    @staticmethod
    def __calc_specificity(tp, fp, tn, fn):
        return tn / max((tn + fp), 0.00000001)


class FlowFrontPredictionEvaluator(Evaluator):
    def __init__(self, name, save_path="/cfs/home/s/c/schroeni/Data/Eval/"):
        self.name = name
        self.save_path = save_path
        self.im_save_path = save_path / "images"
        self.im_save_path.mkdir(parents=True, exist_ok=True)

    def commit(self, inputs, label, *args):
        inputs = np.squeeze(inputs)
        label = np.squeeze(label)
        inp = Image.fromarray(np.uint8(inputs * 255))
        lab = Image.fromarray(np.uint8(label * 255))
        inp.save(self.im_save_path + "inp_" + str(self.name) + ".bmp")
        lab.save(self.im_save_path + "lab_" + str(self.name) + ".bmp")

    def print_metrics(self):
        pass

    def reset(self):
        pass
