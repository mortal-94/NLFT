dataset=NLFT800
base=/kaggle/working/cache_dir/LLM-Research/Meta-Llama-3-8B
data=./data/${dataset}.json
threshold=1.5
gama=0.25
lr=0.00005
num_epochs=10
name=llama3_8b_${dataset}_epoch_${num_epochs}
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune_unlikelihood_dynamic_correct.py \
    --base_model ${base} \
    --data-path ${data} \
    --output_dir ./saved_models/lora/${name} \
    --batch_size 4 \
    --micro_batch_size 1 \
    --num_epochs ${num_epochs} \
    --learning_rate ${lr} \
    --cutoff_len 8192 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --threshold ${threshold}\
    --downsample -1\
    --gama ${gama} \
    --wandb_project "NLFT" \
    --wandb_run_name $name \
    > ${dataset}.txt 2>&1
