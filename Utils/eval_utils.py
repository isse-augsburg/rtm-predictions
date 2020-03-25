import itertools
import os
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import Resources.training as tr_resources


def eval_preparation(save_path):
    """Saves the current repository code and generates a SLURM script to evaluate a trained model more easily.

    Args:
        save_path: directory the trained model is stored in
        abs_file_path: absolute file path to the evaluation script to use for testing
    """

    src_path = Path(sys.argv[0]).parent
    calling_script = Path(sys.argv[0]).parts[-1]
    shutil.copytree(src_path, save_path / "rtm-predictions",
                    ignore=shutil.ignore_patterns('.git*', 'env*', '.idea*', '.vscode*', '__pycache__*',
                                                  'Docs/*', 'Debugging/*', 'Legacy/*'))
    docker_img = 'docker://nvcr.io/isse/pytorch_extended:19.12'
    slurm_txt = f"""#!/bin/sh
#SBATCH --gres=gpu:8
#SBATCH --job-name eval_rtm_predictions
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --mem=495000
#SBATCH --cpus-per-task=75

export SINGULARITY_DOCKER_USERNAME=\\$oauthtoken
export SINGULARITY_DOCKER_PASSWORD={os.getenv('SINGULARITY_DOCKER_PASSWORD')}

""" \
                f'singularity exec --nv -B /cfs:/cfs {docker_img} ' \
                f'python3 -u {save_path}/rtm-predictions/{calling_script} --eval ' \
                f'--eval_path {save_path} ' \
                f'--checkpoint_path {save_path / "checkpoint.pth"} '
    with open(save_path / Path("run_model_eval.sh"), "w") as slurm_script:
        slurm_script.write(slurm_txt)


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
        predicted_non_dryspots = sum(np.array(predictions == 0))
        labels = pred_label[:, 1]
        actual_dryspots = sum(np.array(labels == 1))
        actual_non_dryspots = sum(np.array(labels == 0))
        if consecutive:
            """
            Consecutive means there have to be cons. frames of dryspots, not just the sum of dryspots
            sprinkled all over the run.
            """
            # https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
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
            if actual_dryspots >= threshold_min_counted_dryspots_in_label and predicted_dryspots >= threshold_min_counted_dryspots_in_pred:
                conf_matrix[1][1] += 1
            elif actual_dryspots < threshold_min_counted_dryspots_in_label and predicted_dryspots < threshold_min_counted_dryspots_in_pred:
                conf_matrix[0][0] += 1
            elif actual_dryspots < threshold_min_counted_dryspots_in_label and predicted_dryspots >= threshold_min_counted_dryspots_in_pred:
                conf_matrix[1][0] += 1
            elif actual_dryspots >= threshold_min_counted_dryspots_in_label and predicted_dryspots < threshold_min_counted_dryspots_in_pred:
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
        run_name = k.split('/')[-1:][0]
        plt.title(f"{modelname} - Run {i + 1}")
        plt.legend()
        plt.tight_layout()
        if i == 7 or i == 8:
            plt.savefig(output_dir / f"{modelname.replace(' ', '_')}_run_{i+1}.png")
        # plt.show()
        plt.close()


def get_roc_values_for_different_lengths_of_dryspot_runs():
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


def get_fpr_tpr_thresholds_for_all_vals(input_file: Path):
    with open(input_file, "rb") as f:
        _dict = pickle.load(f)
    pred_label = []
    for k in list(_dict.keys()):
        pred_label.append(np.array(list(_dict[k].values())))
    _all = np.concatenate(pred_label)
    return metrics.roc_curve(y_true=_all[:,1], y_score=_all[:,0])




def get_roc_curves_1140_sensors():
    fpr_1140, tpr_1140, thresholds_1140 = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p")
    fpr_1140d, tpr_1140d, thresholds_1140d = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S1140_densenet_baseline.parent / "advanced_eval/predictions_per_run.p")
    plt.plot(fpr_1140, tpr_1140,
             color='darkorange',
             label=f"1140 Sensors Deconv/Conv\nAUC: {metrics.auc(fpr_1140, tpr_1140):.4f}",
             )
    plt.plot(fpr_1140d, tpr_1140d,
             color='green',
             label=f"1140 Sensors Dense\nAUC: {metrics.auc(fpr_1140d, tpr_1140d):.4f}",
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
    fpr_80, tpr_80, thresholds_80 = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S80_to_ds_thres_longer_train.parent / "advanced_eval/predictions_per_run.p")
    fpr_80_no_pixel_thres, tpr_80_no_pixel_thres, thresholds_80_no_pixel_thres = get_fpr_tpr_thresholds_for_all_vals(
        tr_resources.chkp_S80_to_ds_no_thres.parent / "advanced_eval/predictions_per_run.p")
    fpr_80d, tpr_80d, thresholds_80d = get_fpr_tpr_thresholds_for_all_vals(
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
             label=f"80 Sensors Dense\nAUC: {metrics.auc(fpr_80d, tpr_80d):.4f}",
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
    import pandas as pd
    th_training = pd.read_csv(tr_resources.plots_output/ "data" / "run-2020-03-04_10-41-02_th_0.8_best-tag-Training_Loss.csv")
    th_validation = pd.read_csv(tr_resources.plots_output/ "data" / "run-2020-03-04_10-41-02_th_0.8_best-tag-Validation_Loss.csv")
    no_th_validation= pd.read_csv(tr_resources.plots_output/ "data" / "run-trials_2020-03-05_11-27-46_best_chkp_no_threshold-tag-Validation_Loss.csv")
    no_th_training= pd.read_csv(tr_resources.plots_output/ "data" / "run-trials_2020-03-05_11-27-46_best_chkp_no_threshold-tag-Training_Loss.csv")
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

def __calc_ccc(x, y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = 1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)

    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    return CCC



def calc_ccc_mean(path):
    CCCs = []
    with open(path, "rb") as f:
        _dict = pickle.load(f)
        for k in list(_dict.keys()):
            pred_label = [np.array(list(_dict[k].values()))]
            _all = np.concatenate(pred_label)
            CCC = __calc_ccc(_all[:,0], _all[:,1])
            CCCs.append(CCC)
        return CCCs, np.mean(CCCs)
 
def calc_ccc_global(path):
    pred_label = []
    with open(path, "rb") as f:
        _dict = pickle.load(f)
        for k in list(_dict.keys()):
            pred_label.append(np.array(list(_dict[k].values())))
        _all = np.concatenate(pred_label)
        CCC = __calc_ccc(_all[:,0], _all[:,1])
        return CCC
        


if __name__ == '__main__':
    font = {
        'family': 'normal',
        # 'weight': 'bold',
        'size': 16
    }
    matplotlib.rc('font', **font)

    # plot_trainings()
    # exit()
    #
    # get_roc_curves_1140_sensors()
    # get_roc_curves_80_sensors()
    # exit()
    # get_roc_values_for_different_lengths_of_dryspot_runs()

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
    exit()

    # count_correct_labels_and_predictions(
    #     tr_resources.chkp_S80_to_ds_thres_longer_train.parent / "advanced_eval/predictions_per_run.p", PROB_THRES=.5)
    # for i in np.arange(0.1, 1, 0.1):
    #     count_correct_labels_and_predictions(
    #         tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent / "advanced_eval/predictions_per_run.p",
    #         PROB_THRES=i
    #     )

    # exit()
    fn2 = "consecutive_len_rates_no_overlap.p"
    fn1 = "consecutive_len_rates_overlap.p"
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
