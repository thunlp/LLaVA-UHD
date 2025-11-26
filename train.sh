#!/bin/bash
set -e

##############################
# 分布式参数（单机 8 卡）
##############################
GPUS_PER_NODE=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=12345

DISTRIBUTED_ARGS="\
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT"

echo "Launching with args:"
echo "$DISTRIBUTED_ARGS"

##############################
# 预训练（Stage 1）
##############################
export WANDB_MODE=offline
export TORCH_ELASTIC_RENDEZVOUS_TIMEOUT=10000

MODEL_SETTING="llava-uhd-qwen2-moonvit-so-400m-4-18-se-hirope2d-p10"
DATA_SETTING_STAGE_1="anyres-256-1560-4558k"
DATA_SETTING_STAGE_2="anyres-256-1560-858k"

LLM_CKPT_DIR=/home/guozonghao/user/songhaolin/CKPT/Qwen2-7B-Instruct
CLIP_CKPT_DIR=/home/guozonghao/user/sunshichu/checkpoints/moonvit-so-400m-4-18-se-hirope2d-p10
projector=mlp

BASE_RUN_NAME_STAGE_1="$MODEL_SETTING-$DATA_SETTING_STAGE_1"
BASE_RUN_NAME_STAGE_2="$MODEL_SETTING-$DATA_SETTING_STAGE_2"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME_STAGE_1}"

ACCELERATE_DISABLE_NUMA_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version qwen_2 \
    --model_mode='uhd_v2_5' \
    --data_path path_to_your_data_json \
    --image_folder path_to_your_data_image \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type $projector \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir path_to_your_output_directory \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 6000 \
    --save_total_limit 1 \
    --mm_vision_tower_lr 1e-5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --run_name ${BASE_RUN_NAME_STAGE_1} \
    --attn_implementation flash_attention_2 \
    --single True \
    --resolution 1560 \
    --any_res True \
    --split_patch_size 40 \
    --merger_from_prev False

##############################
# SFT（Stage 2）
##############################
LLM_CKPT_DIR="path_to_your_LLM_path"
DONE_FLAG="$LLM_CKPT_DIR/done.flag"

# 等待 Stage 1 完成
for i in {1..600}; do
    if [ -f "$DONE_FLAG" ]; then
        echo "✅ 检测到 done.flag，预训练完成。"
        break
    fi
    echo "⏳ 等待 done.flag（第 $i 次）"
    sleep 10
done

if [ ! -f "$DONE_FLAG" ]; then
    echo "❌ 超时未检测到 done.flag，终止。"
    exit 1
fi

echo "BASE_RUN_NAME: ${BASE_RUN_NAME_STAGE_2}"

ACCELERATE_DISABLE_NUMA_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version qwen_2 \
    --model_mode='uhd_v2_5' \
    --data_path path_to_your_data_json \
    --image_folder path_to_your_data_image \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type $projector \
    --mm_tunable_parts="mm_vision_tower,mm_language_model,mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir path_to_your_output_directory \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --mm_vision_tower_lr 1e-5 \
    --mm_vision_tower_merger_lr 1e-5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --run_name ${BASE_RUN_NAME_STAGE_2} \
    --attn_implementation flash_attention_2 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --single True \
    --resolution 1560 \
    --any_res True \
    --split_patch_size 40