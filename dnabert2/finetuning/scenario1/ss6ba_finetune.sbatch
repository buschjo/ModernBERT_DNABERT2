#!/bin/bash

#SBATCH --job-name=dna_finetuning
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal
#SBATCH --output=../../../../../sbatch_output/dnabert2_modern/finetuning/dnabert_finetuning/scenario1/%j_%x.out
#SBATCH --error=../../../../../sbatch_output/dnabert2_modern/finetuning/dnabert_finetuning/scenario1/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s0560106@htw-berlin.de

source $HOME/.bashrc
# conda activate ma_dnabert2
# conda activate dnabert2_official_v2
conda activate dnabert2_official_v2_hs

export DATA_PATH=/scratch/jbusch/ma/data
export BASE_MODEL=/scratch/jbusch/ma/models/modern_bert/converted_scenario1/
export MODEL_ID=HF_HS_5M_NRC_150K_199_v6
export TOKENIZER_PATH=/scratch/jbusch/ma/tokenizer/hf_tokenizer/scenario1/HS_5M_NRC
export OUTPUT_PATH=/scratch/jbusch/ma/models/modern_bert/finetuned_scenario1/
export CACHE_DIR=/scratch/jbusch/ma/models/modern_bert/finetuned_scenario1/cache/

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

srun sh ../run_prom_tasks.sh $DATA_PATH $BASE_MODEL $MODEL_ID $TOKENIZER_PATH $OUTPUT_PATH $CACHE_DIR

# Log completion time
echo "Job finished at $(date)"