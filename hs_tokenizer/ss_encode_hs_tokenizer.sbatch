#!/bin/bash

#SBATCH --job-name=hs_encode
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=250G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/bens_tokenizer/encode/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/bens_tokenizer/encode/%j_%x.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate ben_39

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

srun python helix_swap_encode.py \
    --tokenizer /scratch/jbusch/ma/tokenizer/hs_tokenizer/sub3M_tokenizer.json \
    --input_file /scratch/jbusch/ma/data/dnabert_2_pretrain/train.txt \
    --output_file /scratch/jbusch/ma/data/dnabert_2_pretrain/hs_tokenized/sub3M_train.csv

echo "Job finished at $(date)"