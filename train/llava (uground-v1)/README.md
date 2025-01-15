Pretrain:
https://github.com/boyugou/llava_uground/blob/90ff02d24c3f8c7a9fb5c90050fa003b0512910f/scripts/ui_v1/pretrain_7b.sh

SFT:
https://github.com/boyugou/llava_uground/blob/90ff02d24c3f8c7a9fb5c90050fa003b0512910f/scripts/ui_v1/finetune_task_lora.sh

You will likely need to change the dataloader logic a little bit, as I assumed using a parquet file from s3 for data streaming and mistakenly deleted the naive implementation on top of original LLaVA's train.py:
https://github.com/boyugou/llava_uground/blob/90ff02d24c3f8c7a9fb5c90050fa003b0512910f/llava/train/train_s3.py