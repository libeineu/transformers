
TOTAL_NUM_UPDATES=300000
# WARMUP_UPDATES=137
# LR=1e-5
# MAX_SENTENCES=16
# TASK=MRPC

MODEL_DIR=finetuned_BERT_ODE

# --save_strategy no \
# CUDA_VISIBLE_DEVICES=0 python run_glue.py \
#   --model_name_or_path bert-base-cased \
#   --task_name $TASK \
#   --do_train \
#   --do_eval \
#   --adam_beta2 0.98 \
#   --adam_epsilon 1e-6 \
#   --per_device_train_batch_size $MAX_SENTENCES \
#   --learning_rate $LR \
#   --evaluation_strategy epoch \
#   --fp16 \
#   --weight_decay 0.01 \
#   --lr_scheduler_type polynomial \
#   --max_steps $TOTAL_NUM_UPDATES \
#   --warmup_steps $WARMUP_UPDATES \
#   --output_dir $MODEL_DIR \
#   --seed 5 \
#   --overwrite_output_dir \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_mlm.py \
  --model_name_or_path bert-base-cased \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --do_train \
  --do_eval \
  --max_steps $TOTAL_NUM_UPDATES \
  --output_dir $MODEL_DIR