#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2002820
#SBATCH --time=00:60:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1,nvme:10
#SBATCH -e /scratch/project_2002820/lihsin/eb_class/output/%j.err
#SBATCH -o /scratch/project_2002820/lihsin/eb_class/output/%j.out

set -euo pipefail
echo "START: $(date)"

module purge
export SING_IMAGE=$(pwd)/eb_class_latest.sif

BERT_PATH=5832455/0_BERT
cp -r $BERT_PATH $LOCAL_SCRATCH

echo "-------------SCRIPT--------------" >&2
cat $0 >&2
echo -e "\n\n\n" >&2

#srun singularity_wrapper exec \
#singularity shell /path/to/singularity_image.sif

srun singularity exec --nv -B /scratch:/scratch $SING_IMAGE \
    python3 -m finnessayscore.train \
    --epochs 2 \
    --lr 1e-5 \
    --batch_size 16 \
    --grad_acc 1 \
    --model_type seg_essay \
    --jsons data/ismi-kirjoitelmat-parsed.json
    #--use_label_smoothing \
    #--smoothing 0.0 \
    #--whole_essay_overlap 5 \
    #--max_length 512
    #--jsons data/ismi_late_submission-parsed.json
    #--bert_path $BERT_PATH


seff $SLURM_JOBID
echo "END: $(date)"
