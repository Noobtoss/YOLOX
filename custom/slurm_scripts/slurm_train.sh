#!/bin/bash
#SBATCH --job-name=YOLOX_train        # Kurzname des Jobs
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

BASE_DIR=/nfs/scratch/staff/schmittth/codeNexus/YOLOX
CFG=${1:-custom/exps/Images04.py}
CKPT=${2:-models/yolox_x.pth}

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-YOLOX

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb

# python3 setup.py develop

srun python tools/train.py \
    --exp_file $BASE_DIR/$CFG \
    --devices 1 \
    --batch-size 8 \
    --fp16 \
    --occup \
    --ckpt $BASE_DIR/$CKPT \
    --cache \
    --logger wandb \
        wandb-project runs \
        wandb-entity team-noobtoss \
        wandb-name "$(basename "$CKPT" .pth)_$(basename "$CFG" .py)_$(date +"%Y-%m-%d_%H-%M")"
