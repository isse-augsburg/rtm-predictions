import os
import shutil
import sys
from pathlib import Path


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
    docker_img = 'docker://nvcr.io/isse/pytorch_extended:19.11'
    slurm_txt = f"""#!/bin/sh
#SBATCH --gres=gpu:8
#SBATCH --job-name  eval_rtm_predictions
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --mem=495000
#SBATCH --cpus-per-task=75

export SINGULARITY_DOCKER_USERNAME=\\$oauthtoken
export SINGULARITY_DOCKER_PASSWORD={os.getenv('SINGULARITY_DOCKER_PASSWORD')}

""" \
                f'singularity exec --nv -B /cfs:/cfs {docker_img} ' \
                f'python3 -u {save_path}/rtm-predictions/{calling_script} --eval ' \
                f'--eval_path {save_path}'
    with open(save_path / Path("run_model_eval.sh"), "w") as slurm_script:
        slurm_script.write(slurm_txt)
