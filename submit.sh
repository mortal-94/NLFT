#!/bin/bash
base=/root/autodl-tmp/cache_dir/LLM-Research/Meta-Llama-3-8B
name1=llama3_8b_NLFT800_epoch_10/checkpoint_epoch_1
small_name1=NLFT800_checkpoint_epoch_1

CUDA_VISIBLE_DEVICES=0 python merge.py \
    --base_model_name_or_path ${base} \
    --peft_model_path ./saved_models/lora/${name1} \
    --output_dir /root/autodl-tmp/saved_models/${name1}

CUDA_VISIBLE_DEVICES=0 python utils/submit.py \
    --input_json ./data/gsm8k_test_00001.json \
    --llama3_path /root/autodl-tmp/saved_models/${name1} \
    >> $small_name1.log 2>$small_name1.err

rm -r /root/autodl-tmp/saved_models/${name1}
