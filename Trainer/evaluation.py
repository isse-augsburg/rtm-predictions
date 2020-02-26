import itertools
import logging
import os
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas
from PIL import Image

""" 
>>>> PLEASE NOTE: <<<<
Evaluation classes must provide three functions even if not all of them have functionality: 

* commit(output, label, inputs, aux): updates the evaluation state
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
    def __init__(self, save_path: Path = None, sensors_shape=(38, 30), skip_images=True, summary_writer=None):
        super().__init__()
        self.num = 0
        self.save_path = save_path
        self.skip_images = skip_images
        if save_path is not None:
            self.im_save_path = save_path / "images"
            if not self.skip_images:
                self.im_save_path.mkdir(parents=True, exist_ok=True)
        self.sensors_shape = sensors_shape
        self.summary_writer = summary_writer

    def commit(self, net_output, label, inputs, aux, *args):
        if self.skip_images:
            return

        for sample in range(net_output.size()[0]):
            a = net_output[sample].numpy()
            a = np.squeeze(a)
            b = label[sample].numpy()
            b = np.squeeze(b)
            c = inputs[sample].numpy()
            c = np.squeeze(c)
            c = c.reshape(self.sensors_shape[0], self.sensors_shape[1])

            plt.imsave(self.im_save_path / Path(str(self.num) + "out.jpg"), a)
            plt.imsave(self.im_save_path / Path(str(self.num) + "lab.jpg"), b)
            plt.imsave(self.im_save_path / Path(str(self.num) + "inp.jpg"), c)

            self.num += 1
        pass

    def print_metrics(self, step_count):
        pass

    def reset(self):
        self.num = 0
        pass


class BinaryClassificationEvaluator(Evaluator):
    """Evaluator specifically for binary classification. Calculates common metrics and a confusion matrix.
    """

    def __init__(self, save_path: Path = None, skip_images=True, with_text_overlay=False, summary_writer=None):
        super().__init__()
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)
        self.save_path = save_path
        self.skip_images = skip_images
        if save_path is not None:
            self.im_save_path = save_path / "images"
            if not self.skip_images:
                self.im_save_path.mkdir(parents=True, exist_ok=True)
        self.num = 0
        self.with_text_overlay = with_text_overlay
        self.summary_writer = summary_writer
        plt.set_loglevel('warning')

    def commit(self, net_output, label, inputs, aux, *args):
        """Updates the confusion matrix and updates the metrics. 

        Args: 
            net_output: predictions of the model
            label: associated labels
        """
        net_output = net_output.numpy()
        invalid = np.argwhere(np.isnan(net_output[:, 0]))
        if invalid.size > 0:
            invalid = np.reshape(invalid, (invalid.shape[0]))
            net_output = np.delete(net_output, invalid, 0)
            label = np.delete(label, invalid)

        predictions = np.around(net_output[:, 0])

        self.confusion_matrix = np.add(self.confusion_matrix, confusion_matrix(label, predictions, labels=[0, 1]))
        self.tn = self.confusion_matrix[0, 0]
        self.fp = self.confusion_matrix[0, 1]
        self.fn = self.confusion_matrix[1, 0]
        self.tp = self.confusion_matrix[1, 1]

        if not self.skip_images:
            for sample in range(predictions.size):
                c = inputs[sample].numpy()
                c = np.squeeze(c)
                c = c.reshape(143, 111)
                ipred = int(predictions[sample])
                ilabel = int(label[sample])
                if self.with_text_overlay:
                    fig = plt.figure(figsize=(2, 1.55))
                    ax = fig.add_subplot(111)
                    ax.text(45., 75., f'Label={ilabel}\nPred={ipred}', c='red' if ipred != ilabel else 'green')
                    ax.imshow(c)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(self.im_save_path / f"{self.num}-pred_{ipred}_label_{ilabel}.jpg", bbox_inches=extent)
                else:
                    plt.imsave(self.im_save_path / f"{self.num}-pred_{predictions[sample]}_label_{label[sample]}.jpg",
                               c)

        self.num += predictions.size

    def print_metrics(self, step_count=0):
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

        if self.summary_writer is not None:
            acc = self.__calc_accuracy(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)
            prec = self.__calc_precision(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)
            recall = self.__calc_recall(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)
            spec = self.__calc_specificity(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)
            class_names = ["Not OK", "OK"]
            conf_mat_abs = self.__plot_confusion_matrix(self.confusion_matrix, class_names)
            conf_mat_classnorm = self.__plot_confusion_matrix(self.confusion_matrix, class_names, 'class')
            conf_mat_allnorm = self.__plot_confusion_matrix(self.confusion_matrix, class_names, 'all')
            self.summary_writer.add_scalar("Validation/Accuracy", acc, step_count)
            self.summary_writer.add_scalar("Validation/Precision", prec, step_count)
            self.summary_writer.add_scalar("Validation/Recall", recall, step_count)
            self.summary_writer.add_scalar("Validation/Specificity", spec, step_count)
            self.summary_writer.add_figure("Confusion_Matrix/Absolute", conf_mat_abs, step_count)
            self.summary_writer.add_figure("Confusion_Matrix/Normalized_overall", conf_mat_allnorm, step_count)
            self.summary_writer.add_figure("Confusion_Matrix/Normalized_by_class", conf_mat_classnorm, step_count)
        logger.info(f"Accuracy: {acc:7.4f}, Precision: {prec:7.4f}, Recall: {recall:7.4f}, Specificity: {spec:7.4f}")
        df = pandas.DataFrame(self.confusion_matrix, columns=[0, 1], index=[0, 1])
        df = df.rename_axis('Pred', axis=0).rename_axis('True', axis=1)
        logger.info(f'Confusion matrix:\n{df}')

    def reset(self):
        """Resets the internal counters for the next evaluation loop. 
        """

        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.confusion_matrix = np.zeros((2, 2), dtype=int)

    @staticmethod
    def __calc_accuracy(tp, fp, tn, fn):
        return (tp + tn) / max((tp + tn + fp + fn), 1e-8)

    @staticmethod
    def __calc_precision(tp, fp, tn, fn):
        return tp / max((tp + fp), 1e-8)

    @staticmethod
    def __calc_recall(tp, fp, tn, fn):
        return tp / max((tp + fn), 1e-8)

    @staticmethod
    def __calc_specificity(tp, fp, tn, fn):
        return tn / max((tn + fp), 1e-8)

    @staticmethod
    def __plot_confusion_matrix(cm, class_names, norm=''):
        plt.rcParams['figure.constrained_layout.use'] = True
        figure = plt.figure(figsize=(len(class_names) + 1, len(class_names) + 1), dpi=150)

        if norm == 'class':
            cm = np.around(normalize(cm, norm='l1', axis=1), decimals=2)
        elif norm == 'all':
            cm = np.around(cm / max(cm.sum(), 1e-8), decimals=2)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges, vmin=0, vmax=np.sum(cm, 1).max())
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Use white text if squares are dark; otherwise black
        threshold = 0.5 * np.sum(cm, 1).max()
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        return figure
