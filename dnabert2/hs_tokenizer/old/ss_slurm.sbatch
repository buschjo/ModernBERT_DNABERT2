#!/bin/bash

#SBATCH --job-name=cuda_avail
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=2G
#SBATCH --qos=normal
#SBATCH --output=../../sbatch_output/dnabert2_modern/bens_tokenizer/%j_%x.out
#SBATCH --error=../../sbatch_output/dnabert2_modern/bens_tokenizer/%j_%x.err

source $HOME/.bashrc
conda activate ben_39

srun python ./test_cuda.py