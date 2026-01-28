#!/bin/bash
#SBATCH --job-name=YOLOX_train_arr # Kurzname des Jobs
#SBATCH --array=58-64%4              # 7 Jobs total running 4 at a time
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G        # RAM pro CPU Kern #20G #32G #64G

BASE_DIR=/nfs/scratch/staff/schmittth/code-nexus/YOLOX
# CFG=${1:-overrides/exps/Images04.py}
# CKPT=${2:-checkpoints/yolox_x.pth}

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-YOLOX

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_DIR=/tmp/ths_wandb
export WANDB_CACHE_DIR=/tmp/ths_wandb
export WANDB_CONFIG_DIR=/tmp/ths_wandb

PARAMS_FILE="$BASE_DIR/overrides/slurm/slurm_params.txt"
PARAMS=$(grep -v '^[[:space:]]*#' "$PARAMS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

declare -A KV
read -r -a ARR <<< "$PARAMS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
OUTPUT_DIR="${BASE_DIR}/runs"
RUN_NAME="${KV[exp_name]:-unnamed_experiment}"
CFG="${KV[CFG]:-overrides/exps/Images04.py}"
CKPT="${KV[CKPT]:-checkpoints/yolox_x.pth}"

python tools/train.py \
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
        wandb-name "${RUN_NAME}_$(date +"%Y-%m-%d_%H-%M")" \
    output_dir $OUTPUT_DIR \
    $PARAMS

find "$OUTPUT_DIR/$RUN_NAME" -type f ! -name "train_log.txt" ! -name "last_epoch_ckpt.pth" -delete
