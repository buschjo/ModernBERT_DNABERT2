#!/bin/bash

#SBATCH --job-name=hf_dna2_tokenizer
# #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=23G
#SBATCH --qos=normal
#SBATCH --output=../../../../sbatch_output/dnabert2_modern/train_tokenizer/dnabert_tokenizer/%j_%x.out
#SBATCH --error=../../../../sbatch_output/dnabert2_modern/train_tokenizer/dnabert_tokenizer/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate bert24

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

python train_hf_dnabert2_tokenizer.py --data_path /scratch/jbusch/ma/data/dnabert_2_pretrain/tokenizer_training_subsamples/train_sub5M.txt \
    --save_path /scratch/jbusch/ma/tokenizer/hf_tokenizer/ \
    --vocab_size 4096 \
    --tokenizer_name HF_5M_NRC

# Log completion time
echo "Job finished at $(date)"