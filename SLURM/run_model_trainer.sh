#!/bin/sh
#SBATCH --gres=gpu:2
#SBATCH --job-name  ERFH5_Data_model_2 
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --mem=150000
#SBATCH --cpus-per-task=20
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/training-%A-%a.out

# Available containers can be seen at https://ngc.nvidia.com/containers; for credentials contact stieber@isse.de

export SINGULARITY_DOCKER_USERNAME=\$oauthtoken
export SINGULARITY_DOCKER_PASSWORD="get it somewhere!"

singularity exec --nv -B /cfs:/cfs docker://nvcr.io/isse/pytorch_extended:190513 python3 -u  /cfs/home/s/t/stiebesi/Git/tu-kaiserslautern-data/model_trainer.py
