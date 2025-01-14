
TOTAL_NUM_UPDATES=2296
WARMUP_UPDATES=137
LR=2e-5
MAX_SENTENCES=32
TASK=MRPC

MODEL_DIR=../checkpoints/bert_base_LR${LR}_batch${MAX_SENTENCES}_ema_v2

# --save_strategy no \
CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --per_device_train_batch_size $MAX_SENTENCES \
  --learning_rate $LR \
  --evaluation_strategy epoch \
  --fp16 \
  --weight_decay 0.01 \
  --lr_scheduler_type polynomial \
  --max_steps $TOTAL_NUM_UPDATES \
  --warmup_steps $WARMUP_UPDATES \
  --output_dir $MODEL_DIR \
  --seed 5 \
  --overwrite_output_dir \
