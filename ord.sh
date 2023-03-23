#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2004993
#SBATCH --time=00:60:00
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

#BERT_PATH=6539653/0_BERT #5832455/0_BERT
#cp -r $BERT_PATH $LOCAL_SCRATCH

echo "-------------SCRIPT--------------" >&2
cat $0 >&2
echo -e "\n\n\n" >&2

#srun singularity_wrapper exec \
#singularity shell /path/to/singularity_image.sif
srun singularity exec --nv -B /scratch:/scratch $SING_IMAGE \
    python3 -m finnessayscore.train \
    --gpus 1 \
    --epochs 20 \
    --lr 1e-5 \
    --batch_size 16 \
    --grad_acc 3 \
    --model_type trunc_essay \
    --jsons data/ismi-kirjoitelmat-parsed.json \
    --max_length 512
    #--bert_path $BERT_PATH \
    #--use_label_smoothing \
    #--smoothing 0.0 \
    #--whole_essay_overlap 5 \
    #--jsons data/ismi_late_submission-parsed.json



seff $SLURM_JOBID
echo "END: $(date)"
