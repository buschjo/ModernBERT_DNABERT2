#!/bin/bash

#SBATCH --job-name=covid_finetuning
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=../../../../../sbatch_output/dnabert2_modern/finetuning/dnabert_finetuning/scenario2/%j_%x.out
#SBATCH --error=../../../../../sbatch_output/dnabert2_modern/finetuning/dnabert_finetuning/scenario2/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
# conda activate ma_dnabert2
# conda activate dnabert2_official_v2
conda activate dnabert2_official_v2_hs

export DATA_PATH=/scratch/jbusch/ma/data
export BASE_MODEL=/scratch/jbusch/ma/models/modern_bert/converted_scenario2/
export MODEL_ID=HF_HF_2_5M_NRC_150K_199_v10
export TOKENIZER_PATH=/scratch/jbusch/ma/tokenizer/hf_tokenizer/scenario2/hf_dnabert_HF_2_5M_split
export OUTPUT_PATH=/scratch/jbusch/ma/models/modern_bert/finetuned_scenario2/
export CACHE_DIR=/scratch/jbusch/ma/models/modern_bert/finetuned_scenario2/cache/
export FINETUNING_SEED=42

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

srun sh ../run_covid_tasks.sh $DATA_PATH $BASE_MODEL $MODEL_ID $TOKENIZER_PATH $OUTPUT_PATH $CACHE_DIR $FINETUNING_SEED

# Log completion time
echo "Job finished at $(date)"