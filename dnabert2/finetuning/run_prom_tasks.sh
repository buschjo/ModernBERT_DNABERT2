#!/bin/bash
# adapted from https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/scripts/run_dnabert2.sh (accessed: 01.05.2025)

data_path=$1
scratch=/scratch/jbusch/ma/
# base_model="zhihan1996/DNABERT-2-117M"
# base_model=${scratch}models/modern_bert/converted_to_hf/hs_tokenizer5M_v5
base_model=$2
model_id=$3
# tokenizer_path=${scratch}tokenizer/hf_tokenizer/hf_dnabert_sub5M
# tokenizer_path=${scratch}models/modern_bert/converted_to_hf/test_conversion
# tokenizer_path=${scratch}tokenizer/hs_tokenizer/fixed_encode_decode/sub5M_tokenizer
tokenizer_path=$4
# tokenizer_path="zhihan1996/DNABERT-2-117M"
# output_path=${scratch}models/modern_bert/finetuning/
output_path=$5
# cache_dir=${scratch}models/finetuning/cache/
cache_dir=$6
lr=3e-5

echo "The provided data_path is $data_path"

for seed in 42
do
    for data in prom_core_all prom_core_notata
    do
        python ../dnabert2_finetuning.py \
            --model_name_or_path ${base_model}${model_id} \
			--tokenizer_path ${tokenizer_path} \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_id}_${lr}_prom_${data}_seed${seed} \
			--model_max_length 70 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir ${output_path}${model_id} \
			--cache_dir ${cache_dir}${model_id} \
            --eval_strategy epoch \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
			--save_strategy epoch \
            --find_unused_parameters False
    done

    for data in prom_core_tata
    do
        python ../dnabert2_finetuning.py \
            --model_name_or_path ${base_model}${model_id} \
			--tokenizer_path ${tokenizer_path} \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_id}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
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
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python ../dnabert2_finetuning.py \
            --model_name_or_path ${base_model}${model_id} \
			--tokenizer_path ${tokenizer_path} \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_id}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 300 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir ${output_path}${model_id} \
			--cache_dir ${cache_dir}${model_id} \
            --eval_strategy epoch \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
			--save_strategy epoch \
            --find_unused_parameters False
    done

    for data in prom_300_tata
    do 
        python ../dnabert2_finetuning.py \
            --model_name_or_path ${base_model}${model_id} \
			--tokenizer_path ${tokenizer_path} \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name ${model_id}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 300 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
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
            --find_unused_parameters False
    done 
done