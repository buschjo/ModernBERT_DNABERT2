#!/bin/bash

#SBATCH --job-name=local_tok
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/local_tokenizer/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/local_tokenizer/%j_%x.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate bert24

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

composer main.py yamls/main/jb_dna_mosaic_local_tokenizer-bert-base-uncased.yaml

# Log completion time
echo "Job finished at $(date)"