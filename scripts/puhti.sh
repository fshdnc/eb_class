#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2004993
#SBATCH --time=00:60:00
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

lr="$1"
ga="$2"

#srun singularity_wrapper exec \
#singularity shell /path/to/singularity_image.sif
#whole essay, 20e, lr 1e-5, batch 1, grad_acc 16, overlapping token 5, cross-entropy, max token 512
srun singularity exec --nv -B /scratch:/scratch $SING_IMAGE \
    python3 -m finnessayscore.train \
    --epochs 20 \
    --lr $lr \
    --batch_size 8 \
    --grad_acc $ga \
    --model_type trunc_essay \
    --pooling mean \
    --jsons data/ismi-kirjoitelmat-parsed-nopunc.json
    #--whole_essay_overlap 5 \
    #--use_label_smoothing \
    #--smoothing 0.0 \
    #--max_length 512
    #--jsons data/ismi_late_submission-parsed.json
    #--bert_path $BERT_PATH


seff $SLURM_JOBID
echo "END: $(date)"
