#!/bin/bash

#SBATCH --job-name=hs_train_encode
#SBATCH --nodes=1
##SBATCH --nodelist=kiwinode02
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/bens_tokenizer/encode/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/bens_tokenizer/encode/%j_%x.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
# conda activate ben_39
conda activate bens_tokenizer_modernbert2

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

srun python helix_swap_train_encode.py \
    --tokenizer_training_data /scratch/jbusch/ma/data/dnabert_2_pretrain/train_sub1k.txt \
    --tokenizer_save_path /scratch/jbusch/ma/tokenizer/hs_tokenizer/sub1k_tokenizer.json \
    --pretraining_data /scratch/jbusch/ma/data/dnabert_2_pretrain/train.txt \
    --output_tokenized /scratch/jbusch/ma/data/dnabert_2_pretrain/hs_tokenized/sub1k_train.csv

echo "Job finished at $(date)"