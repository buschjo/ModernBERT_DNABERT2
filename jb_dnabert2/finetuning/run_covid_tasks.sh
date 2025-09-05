#!/bin/bash
# adapted from https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/scripts/run_dnabert2.sh (accessed: 05.05.2025)

data_path=$1
scratch=/scratch/jbusch/ma/
base_model=$2
model_id=$3
tokenizer_path=$4
output_path=$5
cache_dir=$6
lr=3e-5
finetuning_seed=$7

echo "The provided data_path is $data_path"

for seed in ${finetuning_seed}
do
	for data in covid
   	do
        python ../dnabert2_finetuning.py  \
            --model_name_or_path ${base_model}${model_id} \
			--tokenizer_path ${tokenizer_path} \
            --data_path  $data_path/GUE/virus/$data \
            --kmer -1 \
            --run_name ${model_id}_${lr}_virus_${data}_seed${seed} \
            --model_max_length 256 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 8 \
            --fp16 \
            --save_steps 200 \
            --output_dir ${output_path}${model_id} \
			--cache_dir ${cache_dir}${model_id} \
            --eval_strategy epoch \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
			--save_strategy epoch \
            --find_unused_parameters False \
			--seed ${finetuning_seed}
    done
done