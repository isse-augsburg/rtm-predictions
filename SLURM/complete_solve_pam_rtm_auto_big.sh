#!/bin/sh
#SBATCH --partition=big-cpu
#SBATCH --mem=24000
#SBATCH --time=1000:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task=32
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%a.out
#SBATCH --array=1-0%9

srun -t 15 singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np 32 /cfs/share/data/RTM/Lautern/output/with_shapes/2019-05-16_14-16-04_1p/${SLURM_ARRAY_TASK_ID}/2019-05-16_14-16-04_${SLURM_ARRAY_TASK_ID}g.unf
