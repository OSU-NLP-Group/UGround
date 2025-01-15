#!/bin/bash

torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port $MASTER_PORT \
      finetune/finetune.py \
    --model_name_or_path Qwen/Qwen-VL-Chat\
    --qwen_path Qwen/Qwen-VL-Chat \
    --data_path ./data/filtered_data.json  \
    --bf16 True \
    --fix_vit False \
    --output_dir ./checkpoint/ \
    --num_train_epochs 0.91 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 30 \
    --learning_rate 3e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --use_lora \
    --bf16 True \
    --gradient_checkpointing \
    --deepspeed finetune/ds_config_zero2.json