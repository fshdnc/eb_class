#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2004993
#SBATCH --time=00:60:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1,nvme:10
#SBATCH -e /scratch/project_2004993/li/eb_class/output/%j.err
#SBATCH -o /scratch/project_2004993/li/eb_class/output/%j.out

set -euo pipefail
echo "START: $(date)"

module purge
export SING_IMAGE=/scratch/project_2004993/sifs/eb_class_latest.sif

echo "-------------SCRIPT--------------" >&2
cat $0 >&2
echo -e "\n\n\n" >&2

srun singularity exec --nv -B /scratch:/scratch $SING_IMAGE \
    python3 -m finnessayscore.train \
    --gpus 1 \
    --epochs 1 \
    --lr 1e-5 \
    --batch_size 16 \
    --grad_acc 3 \
    --model_type trunc_essay \
    --data_dir data/ismi \
    --max_length 512
    #--bert_path $BERT_PATH \
    #--use_label_smoothing \
    #--smoothing 0.0 \
    #--whole_essay_overlap 5

seff $SLURM_JOBID
echo "END: $(date)"
