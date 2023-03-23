#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2004993
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -e /scratch/project_2004993/lihsin/eb_class/output/%j.err
#SBATCH -o /scratch/project_2004993/lihsin/eb_class/output/%j.out

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
    python3 -m finnessayscore.explain_trunc \
    --batch_size 1 \
    --model_type trunc_essay \
    --max_length 512 \
    --load_checkpoint lightning_logs/2021-09-23_105120.806734/latest/checkpoints/best_trunc_mean_momaf.ckpt \
    --pooling mean \
    --class_nums data/momaf.pickle \
    --jsons data/momaf-masked.json

#data/sentiment.json
#sent_trunc_essay_mean_7869468.ckpt \
#best_trunc_essay_mean_7611708.ckpt \
#momaf_trunc_essay_mean_7870048.ckpt \
#data/ismi-kirjoitelmat-parsed.json

seff $SLURM_JOBID
echo "END: $(date)"
