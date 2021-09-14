#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2002820
#SBATCH --time=01:10:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -e /scratch/project_2002820/lihsin/eb_class/output/%j.err
#SBATCH -o /scratch/project_2002820/lihsin/eb_class/output/%j.out

set -euo pipefail
echo "START: $(date)"

module purge
export SING_IMAGE=$(pwd)/eb_class_latest.sif
export TRANSFORMERS_CACHE=$(realpath cache)

#BERT_PATH=5832455/0_BERT
#cp -r $BERT_PATH $LOCAL_SCRATCH

echo "-------------SCRIPT--------------" >&2
cat $0 >&2
echo -e "\n\n\n" >&2

srun singularity exec --nv -B /scratch:/scratch $SING_IMAGE \
    python3 -m finnessayscore.explain_trunc\
    --batch_size 1 \
    --model_type trunc_essay \
    --max_length 512 \
    --load_checkpoint  best_trunc_essay_7502465.ckpt \
    --jsons data/ismi-kirjoitelmat-parsed.json


seff $SLURM_JOBID
echo "END: $(date)"
