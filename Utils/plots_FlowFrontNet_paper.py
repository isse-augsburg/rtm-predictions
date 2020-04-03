import itertools
import pickle
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score

import Resources.training as tr_resources


def count_correct_labels_and_predictions(input_file: Path,
                                         threshold_min_counted_dryspots_in_label=5,
                                         threshold_min_counted_dryspots_in_pred=5,
                                         PROB_THRES=0.5,
                                         consecutive=False,
                                         min_len_consecutive=10,
                                         overlap=False):
    conf_matrix = np.zeros((2, 2), dtype=int)
    np.set_printoptions(precision=3, suppress=True)
    with open(input_file, "rb") as f:
        _dict = pickle.load(f)
    runs_labels = []
    runs_preds = []
    for k in list(_dict.keys()):
        one_run = _dict[k]
        pred_label = np.asarray([one_run[k] for k in one_run], dtype=float)
        predictions = pred_label[:, 0]
        predicted_dryspots = sum(np.array(predictions > PROB_THRES))
        # predicted_non_dryspots = sum(np.array(predictions == 0))
        labels = pred_label[:, 1]
        actual_dryspots = sum(np.array(labels == 1))
        # actual_non_dryspots = sum(np.array(labels == 0))
        if consecutive:
            """
            Consecutive means there have to be cons. frames of dryspots, not just the sum of dryspots
            sprinkled all over the run.
            """
            # https://stackoverflow.com/questions/24342047/
            # count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
            lens_of_runs_of_pred_dryspots = [sum(1 for _ in group) for key, group in
                                             itertools.groupby(np.array(predictions > PROB_THRES)) if key]
            lens_of_runs_of_actual_dryspots = [sum(1 for _ in group) for key, group in
                                               itertools.groupby(np.array(labels == 1)) if key]
            max_len_pred = 0 if len(lens_of_runs_of_pred_dryspots) == 0 else max(lens_of_runs_of_pred_dryspots)
            max_len_actual = 0 if len(lens_of_runs_of_actual_dryspots) == 0 else max(lens_of_runs_of_actual_dryspots)
            overlap_of_consecutive_dryspots = np.logical_and(np.array(labels == 1), np.array(predictions > PROB_THRES))
            len_overlap_dryspots = [sum(1 for _ in group) for key, group in
                                    itertools.groupby(np.array(overlap_of_consecutive_dryspots == 1)) if key]
            if overlap:
                """
                Overlap is even stricter than `consecutive`: Here, it is necessary for the predicted dryspots to be at 
                the same spots in the run as the label. The longest run of dryspots inside one run is counted.
                """
                max_len_pred = 0 if len(len_overlap_dryspots) == 0 else max(len_overlap_dryspots)
            if max_len_actual >= min_len_consecutive and max_len_pred >= min_len_consecutive:
                runs_labels.append(1)
                runs_preds.append(1)
                conf_matrix[1][1] += 1
            elif max_len_actual < min_len_consecutive and max_len_pred < min_len_consecutive:
                runs_labels.append(0)
                runs_preds.append(0)
                conf_matrix[0][0] += 1
            elif max_len_actual < min_len_consecutive and max_len_pred >= min_len_consecutive:
                # False positive
                runs_labels.append(0)
                runs_preds.append(1)
                conf_matrix[1][0] += 1
            elif max_len_actual >= min_len_consecutive and max_len_pred < min_len_consecutive:
                # False negative
                runs_labels.append(1)
                runs_preds.append(0)
                conf_matrix[0][1] += 1
        else:
            if actual_dryspots >= threshold_min_counted_dryspots_in_label and \
                    predicted_dryspots >= threshold_min_counted_dryspots_in_pred:
                conf_matrix[1][1] += 1
            elif actual_dryspots < threshold_min_counted_dryspots_in_label and \
                    predicted_dryspots < threshold_min_counted_dryspots_in_pred:
                conf_matrix[0][0] += 1
            elif actual_dryspots < threshold_min_counted_dryspots_in_label and \
                    predicted_dryspots >= threshold_min_counted_dryspots_in_pred:
                conf_matrix[1][0] += 1
            elif actual_dryspots >= threshold_min_counted_dryspots_in_label and \
                    predicted_dryspots < threshold_min_counted_dryspots_in_pred:
                conf_matrix[0][1] += 1
    if consecutive:
        fpr, tpr, thresholds = metrics.roc_curve(np.array(runs_labels), np.array(runs_preds),
                                                 pos_label=None, sample_weight=None, drop_intermediate=True)
        print(f"Thres: {PROB_THRES}, Min consecutive Dryspots p. run: {min_len_consecutive},"
              f" Acc: {(conf_matrix[0][0] + conf_matrix[1][1]) / conf_matrix.sum()}")
    else:
        print(f"Thres: {PROB_THRES}, Min Counted Dryspots p. run/label: {threshold_min_counted_dryspots_in_label},"
              f" Min Counted Dryspots p. run/prediction: {threshold_min_counted_dryspots_in_pred},"
              f" Acc: {(conf_matrix[0][0] + conf_matrix[1][1]) / conf_matrix.sum()}")
    print(conf_matrix)
    return fpr, tpr


def plot_labels_and_predictions_per_run(modelname, input_file: Path, output_dir: Path, num_runs=10):
    output_dir.mkdir(exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)
    with open(input_file, "rb") as f:
        _dict = pickle.load(f)
    for i, k in enumerate(list(_dict.keys())[:num_runs]):
        one_run = _dict[k]
        pred_label = np.asarray([one_run[k] for k in one_run], dtype=float)
        predictions = pred_label[:, 0]
        labels = pred_label[:, 1]
        plt.plot(range(len(labels)), predictions, label="prediction")
        plt.plot(range(len(labels)), labels, label="label")
        plt.xlabel("Steps")
        plt.ylabel("Dry Spot")
        plt.ylim((-0.1, 1.1))
        # run_name = k.split('/')[-1:][0]
        plt.title(f"{modelname} - Run {i + 1}")
        plt.legend()
        plt.tight_layout()
        if i == 7 or i == 8:
            plt.savefig(output_dir / f"{modelname.replace(' ', '_')}_run_{i + 1}.png")
        # plt.show()
        plt.close()


def plot_labels_and_predictions_of_three_models_per_run(names: list, input_files: list, output_dir: Path, num_runs=10):
    output_dir.mkdir(exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)
    data_dicts = []
    for input_file in input_files:
        with open(input_file, "rb") as f:
            data_dicts.append(pickle.load(f))
    for i, k in enumerate(list(data_dicts[0].keys())[:num_runs][7:9]):
        one_run_mod0 = data_dicts[0][k]
        one_run_mod1 = data_dicts[1][k]
        one_run_mod2 = data_dicts[2][k]
        labels = np.asarray([one_run_mod0[k] for k in one_run_mod0], dtype=float)[:, 1]
        predictions0 = np.asarray([one_run_mod0[k] for k in one_run_mod0], dtype=float)[:, 0]
        predictions1 = np.asarray([one_run_mod1[k] for k in one_run_mod1], dtype=float)[:, 0]
        predictions2 = np.asarray([one_run_mod2[k] for k in one_run_mod2], dtype=float)[:, 0]

        plt.plot(range(len(labels)), predictions1, label=f"Pred. {names[1]}", linestyle='-.')
        plt.plot(range(len(labels)), predictions2, label=f"Pred. {names[2]}", linestyle='--')
        plt.plot(range(len(labels)), predictions0, label=f"Pred. {names[0]}", linestyle=':')
        plt.plot(range(len(labels)), labels, label="Label")
        plt.xlabel("Steps")
        plt.ylabel("Dry Spot")
        plt.ylim((-0.1, 1.1))
        # run_name = k.split('/')[-1:][0]
        plt.title(f"All Models - Run {i + 8}")
        plt.legend()
        plt.tight_layout()
        # if i == 7 or i == 8:
        plt.savefig(output_dir / f"run_{i + 8}.png")
        # plt.show()
        plt.close()


def get_roc_values_for_different_lengths_of_dryspot_runs():
    """
    Testing different dryspot run lengths
    """
    print(tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p")
    rates = []
    for i in np.arange(0, 1.1, 0.1):  # list(range(0, 200, 10))[1:]:
        fpr, tpr = count_correct_labels_and_predictions(
            tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p",
            PROB_THRES=i,
            threshold_min_counted_dryspots_in_label=5,  # Only useful when using non-consecutive dryspots
            threshold_min_counted_dryspots_in_pred=5,  # Only useful when using non-consecutive dryspots
            consecutive=True,
            min_len_consecutive=1,
            overlap=False
        )
        rates.append((fpr, tpr, i))
    pickle.dump(rates, open("consecutive_len_rates_no_overlap_different_thres.p", "wb"))


def wrapper_acc(y_score, y_true, thresh):
    score = np.where(y_score > thresh, 1, 0)
    acc = accuracy_score(y_true, score)
    acc_thresh = (acc, thresh)
    return acc_thresh


def get_fpr_tpr_thresholds_for_all_vals(input_file: Path):
    with open(input_file, "rb") as f:
        _dict = pickle.load(f)
    pred_label = []
    for k in list(_dict.keys()):
        pred_label.append(np.array(list(_dict[k].values())))
    _all = np.concatenate(pred_label)
    y_true = _all[:, 1]
    y_score = _all[:, 0]
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score)
    with Pool() as p:
        accs_threshs = p.map(partial(wrapper_acc, y_score, y_true), np.arange(0., 1, 0.01))

    accs, threshs = zip(*accs_threshs)
    max_acc = np.array(accs).max()
    index = np.array(accs).argmax()
    return fpr, tpr, thresholds, (max_acc, threshs[index], accs)


def get_max_acc_for_certain_thresholds(p: Path):
    fpr, tpr, thresholds, (max_acc, thresh_max_acc, accs) = get_fpr_tpr_thresholds_for_all_vals(p)
    print(max_acc, thresh_max_acc)
    # with open("max_acc_thresh_accs_1140.p", "wb") as f:
    #     pickle.dump((max_acc, thresh_max_acc, accs), f)


def get_roc_curves_1140_sensors():
    fpr_1140, tpr_1140, thresholds_1140, _ = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p")

    fpr_1140d, tpr_1140d, thresholds_1140d, _ = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S1140_densenet_baseline_full_trainingset.parent / "advanced_eval/predictions_per_run.p")
    plt.plot(fpr_1140, tpr_1140,
             color='darkorange',
             label=f"1140 Sensors Deconv/Conv\nAUC: {metrics.auc(fpr_1140, tpr_1140):.4f}",
             )
    plt.plot(fpr_1140d, tpr_1140d,
             color='green',
             label=f"1140 Sensors Feed Forward\nAUC: {metrics.auc(fpr_1140d, tpr_1140d):.4f}",
             linestyle="-.")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(tr_resources.plots_output / "ROC_curves_1140.png")
    # plt.show()
    plt.close()


def get_roc_curves_80_sensors():
    fpr_80, tpr_80, thresholds_80, _ = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S80_to_ds_thres_longer_train.parent / "advanced_eval/predictions_per_run.p")
    fpr_80_no_pixel_thres, tpr_80_no_pixel_thres, thresholds_80_no_pixel_thres, _ = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S80_to_ds_no_thres.parent / "advanced_eval/predictions_per_run.p")
    fpr_80d, tpr_80d, thresholds_80d, _ = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S80_densenet_baseline.parent / "advanced_eval/predictions_per_run.p")

    plt.plot(fpr_80, tpr_80,
             color='red',
             label=f"80 Sensors Deconv/Conv *\nAUC: {metrics.auc(fpr_80, tpr_80):.4f}",
             )
    plt.plot(fpr_80_no_pixel_thres, tpr_80_no_pixel_thres,
             color='black',
             label=f"80 Sensors Deconv/Conv\n"
                   f"AUC: {metrics.auc(fpr_80_no_pixel_thres, tpr_80_no_pixel_thres):.4f}",
             linestyle="-.")
    plt.plot(fpr_80d, tpr_80d,
             color='brown',
             label=f"80 Sensors Feed Forward\nAUC: {metrics.auc(fpr_80d, tpr_80d):.4f}",
             linestyle=":")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(tr_resources.plots_output / "ROC_curves_80.png")
    # plt.show()
    plt.close()


def plot_trainings():
    th_training = pd.read_csv(tr_resources.plots_output / "data" /
                              "run-2020-03-04_10-41-02_th_0.8_best-tag-Training_Loss.csv")
    th_validation = pd.read_csv(tr_resources.plots_output / "data" /
                                "run-2020-03-04_10-41-02_th_0.8_best-tag-Validation_Loss.csv")
    no_th_validation = pd.read_csv(tr_resources.plots_output / "data" /
                                   "run-trials_2020-03-05_11-27-46_best_chkp_no_threshold-tag-Validation_Loss.csv")
    no_th_training = pd.read_csv(tr_resources.plots_output / "data" /
                                 "run-trials_2020-03-05_11-27-46_best_chkp_no_threshold-tag-Training_Loss.csv")
    time_steps_no_th = no_th_training["Wall time"] - no_th_training["Wall time"][0]
    time_steps_th = th_training["Wall time"] - th_training["Wall time"][0]
    plt.plot(np.asarray(time_steps_no_th), np.asarray(no_th_training["Value"]), label="Not thresholded")
    plt.plot(np.asarray(time_steps_th), np.asarray(th_training["Value"]), label="Thresholded at .8")
    plt.legend()
    plt.ylabel("Training Loss")
    plt.xlabel("Time in Seconds")
    plt.grid()
    plt.tight_layout()
    plt.savefig(tr_resources.plots_output / "Training_loss.png")
    # plt.show()
    plt.close()
    time_steps_no_th_v = no_th_validation["Wall time"] - no_th_validation["Wall time"][0]
    time_steps_th_v = th_validation["Wall time"] - th_validation["Wall time"][0]
    plt.plot(np.asarray(time_steps_no_th_v), np.asarray(no_th_validation["Value"]), label="Not thresholded")
    plt.plot(np.asarray(time_steps_th_v), np.asarray(th_validation["Value"]), label="Thresholded at .8")
    plt.ylabel("Validation Loss")
    plt.xlabel("Time in Seconds")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(tr_resources.plots_output / "Validation_loss.png")
    # plt.show()
    plt.close()
    # no_th_validation["Value"].plot(label="Validation Loss - Not thresholded")
    # th_validation["Value"].plot(label="Validation Loss - Thresholded at .8")
    # plt.legend()
    # plt.grid()
    # plt.xlim((-0.1, 4.1))
    # plt.show()
    # plt.close()


def print_rel_conf_matrix_at_certain_threshold(input_file: Path, threshold: float):
    np.set_printoptions(precision=2)
    with open(input_file, "rb") as f:
        _dict = pickle.load(f)
    pred_label = []
    for k in list(_dict.keys()):
        pred_label.append(np.array(list(_dict[k].values())))
    _all = np.concatenate(pred_label)
    y_true = _all[:, 1]
    y_score = _all[:, 0]
    print(input_file.parent.parent.stem, f"Threshold: {threshold}")
    yy_score = np.where(y_score > threshold, 1, 0)
    cm = metrics.confusion_matrix(y_true, yy_score, normalize="all")
    print(cm * 100)
    print("Ref: Threshold 0.5")
    y_score = np.where(y_score > .5, 1, 0)
    cm = metrics.confusion_matrix(y_true, y_score, normalize="all")
    print(cm * 100)


def plot_preds_and_labels(all_in_one=True):
    if all_in_one:
        plot_labels_and_predictions_of_three_models_per_run(
            ["S80 Pixel Threshold", "S1140", "S80"],
            [tr_resources.chkp_S80_to_ds_thres_longer_train.parent / "advanced_eval/predictions_per_run.p",
             tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p",
             tr_resources.chkp_S80_to_ds_no_thres.parent / "advanced_eval/predictions_per_run.p"],
            output_dir=tr_resources.plots_output
        )
    else:
        plot_labels_and_predictions_per_run(
            "S80 Pixel Threshold",
            tr_resources.chkp_S80_to_ds_thres_longer_train.parent / "advanced_eval/predictions_per_run.p",
            output_dir=tr_resources.plots_output
        )
        plot_labels_and_predictions_per_run(
            "S1140",
            tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p",
            output_dir=tr_resources.plots_output
        )

        plot_labels_and_predictions_per_run(
            "S80",
            tr_resources.chkp_S80_to_ds_no_thres.parent / "advanced_eval/predictions_per_run.p",
            output_dir=tr_resources.plots_output
        )


def get_max_acc_for_certain_thresholds_for_many_models():
    # 0.7522373199462891 0.51
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S20_to_ds_retrain.parent /
                                       "advanced_eval/predictions_per_run.p")
    # 0.8148288726806641 0.58
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S80_to_ds_no_thres.parent /
                                       "advanced_eval/predictions_per_run.p")
    # 0.9168109893798828 0.49
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent /
                                       "advanced_eval/predictions_per_run.p")
    # 0.8369369506835938 0.57
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S80_to_ds_thres_longer_train.parent /
                                       "advanced_eval/predictions_per_run.p")
    # 0.8274965286254883 0.54
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S1140_densenet_baseline_full_trainingset.parent /
                                       "advanced_eval/predictions_per_run.p")
    # 0.7957048416137695 0.52
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S80_densenet_baseline.parent /
                                       "advanced_eval/predictions_per_run.p")
    # 0.7468843460083008 0.49
    get_max_acc_for_certain_thresholds(tr_resources.chkp_S20_densenet_baseline_full_trainingset.parent /
                                       "advanced_eval/predictions_per_run.p")


def relative_conf_matrices_for_models():
    print_rel_conf_matrix_at_certain_threshold(tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent /
                                               "advanced_eval/predictions_per_run.p", 0.49)
    print_rel_conf_matrix_at_certain_threshold(tr_resources.chkp_S1140_densenet_baseline_full_trainingset.parent /
                                               "advanced_eval/predictions_per_run.p", 0.54)
    print_rel_conf_matrix_at_certain_threshold(tr_resources.chkp_S80_to_ds_thres_longer_train.parent /
                                               "advanced_eval/predictions_per_run.p", 0.57)
    print_rel_conf_matrix_at_certain_threshold(tr_resources.chkp_S80_densenet_baseline.parent /
                                               "advanced_eval/predictions_per_run.p", 0.52)


def get_roc_curves():
    get_roc_curves_1140_sensors()
    get_roc_curves_80_sensors()


def different_tpr_fpr_with_and_without_overlap():
    fn2 = "consecutive_len_rates_no_overlap.p"
    # fn1 = "consecutive_len_rates_overlap.p"
    fn = "consecutive_len_rates_no_overlap_different_thres.p"
    with open(fn2, "rb") as f:
        rates = pickle.load(f)
    fpr, tpr, thresholds = zip(*rates[:-1:])
    _fpr = [x[1] for x in fpr]
    _tpr = [x[1] for x in tpr]
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(_fpr, _tpr, color='darkorange')
    plt.scatter(_fpr, _tpr, color='red')
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    for i, txt in enumerate(thresholds):
        plt.annotate(f"{txt}" if type(txt) == int else f"{txt:.2f}", (_fpr[i], _tpr[i]))
    plt.grid()
    plt.title(fn)
    plt.show()


if __name__ == '__main__':
    font = {
        'family': 'normal',
        # 'weight': 'bold',
        'size': 16
    }
    matplotlib.rc('font', **font)
    get_roc_curves()
