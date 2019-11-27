import os
import shutil
from pathlib import Path


def eval_preparation(save_path, abs_file_path):
    """Saves the current repository code and generates a SLURM script to evaluate a trained model more easily.

    Args:
        save_path: directory the trained model is stored in
        abs_file_path: absolute file path to the evaluation script to use for testing
    """

    src_path, filename = os.path.split(abs_file_path)
    shutil.copytree(src_path, save_path / "code", ignore=shutil.ignore_patterns('.git*', 'env*', '.idea*',
                                                                                '.vscode*', '__pycache__*'))

    slurm_txt = """#!/bin/sh
    #SBATCH --gres=gpu:8
    #SBATCH --job-name  rtm_predictions_eval
    #SBATCH --ntasks=1
    #SBATCH -p gpu
    #SBATCH --mem=150000
    #SBATCH --cpus-per-task=20

    export SINGULARITY_DOCKER_USERNAME=\\$oauthtoken
    export SINGULARITY_DOCKER_PASSWORD=[insert key]

    singularity exec --nv -B /cfs:/cfs docker://nvcr.io/isse/pytorch_extended:19.10 python3"""
    slurm_txt += f" -u  {save_path}/code/{filename} --eval --eval_path {save_path}\n"
    with open(save_path / Path("run_model_eval.sh"), "w") as slurm_script:
        slurm_script.write(slurm_txt)
