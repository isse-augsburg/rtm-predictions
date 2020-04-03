import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

import Resources.training as tr_resources
from Trainer.evaluation import BinaryClassificationEvaluator


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


def run_eval_w_binary_classificator(output_dir, modeltrainer, chkp_p: Path, ):
    output_dir.mkdir(exist_ok=True)
    modeltrainer.inference_on_test_set(
        output_path=output_dir,
        checkpoint_path=chkp_p,
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(output_dir,
                                      skip_images=True,
                                      with_text_overlay=True,
                                      advanced_eval=True)
    )
    with open(output_dir / "predictions_per_run.p", "wb") as f:
        pickle.dump(modeltrainer.classification_evaluator.origin_tracker, f)


def __calc_ccc(x, y):
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    x_var = 1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)
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
            CCC = __calc_ccc(_all[:, 0], _all[:, 1])
            if CCC == 0:
                print('x')
            CCCs.append(CCC)
        return CCCs, np.mean(CCCs)


def calc_ccc_global(path):
    pred_label = []
    with open(path, "rb") as f:
        _dict = pickle.load(f)
        for k in list(_dict.keys()):
            pred_label.append(np.array(list(_dict[k].values())))
        _all = np.concatenate(pred_label)
        CCC = __calc_ccc(_all[:, 0], _all[:, 1])
        return CCC


def run_ccc_calculations():
    ccc_glob = calc_ccc_global(tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent /
                               "advanced_eval/predictions_per_run.p")
    ccc_mean = calc_ccc_mean(tr_resources.chkp_S1140_to_ds_0_basepr_frozen.parent /
                             "advanced_eval/predictions_per_run.p")
    print(ccc_glob, ccc_mean[1])
