#!/bin/bash

#SBATCH --job-name=train_sentencePiece
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=493G
#SBATCH --qos=normal
#SBATCH --output=../../../sbatch_output/dnabert2_modern/train_tokenizer/%j_%x.out
#SBATCH --error=../../../sbatch_output/dnabert2_modern/train_tokenizer/%j_%x.err

##SBATCH --mail-type=ALL
##SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
conda activate standarsentencePiece

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

python train_sp_tokenizer.py

# Log completion time
echo "Job finished at $(date)"