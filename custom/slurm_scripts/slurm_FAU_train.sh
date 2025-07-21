#!/bin/bash -l
#SBATCH --job-name=YOLOX_train
#SBATCH --output=logs/R-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

BASE_DIR="$WORK/codeNexus/YOLOX"

module purge                  # Purge any pre-existing modules
module load python/3.12-conda # Load Python/Conda module
module load cuda/12.6.1       # Load CUDA module for GPU

conda activate conda_YOLOX

# python3 setup.py develop
# TODO