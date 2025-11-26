#!/bin/bash
set -e

export WANDB_MODE=offline
export TORCH_ELASTIC_RENDEZVOUS_TIMEOUT=10000

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
WORLD_SIZE=${SLURM_NNODES:-1}
RANK=${SLURM_PROCID:-0}

if command -v scontrol &> /dev/null && [ -n "$SLURM_NODELIST" ]; then
    MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
else
    MASTER_ADDR="localhost"
fi

MASTER_PORT=12345

DISTRIBUTED_ARGS="\
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT"

echo "Launching with args:"
echo "$DISTRIBUTED_ARGS"

############### Pretrain ################

MODEL_SETTING="llava-uhd-qwen3-moonvit-so-400m-4-18-se-hirope2d-p10"
DATA_SETTING_STAGE_1="anyres-256-1560-4558k"
DATA_SETTING_STAGE_2="anyres-256-1560-ocr-inter"
DATA_SETTING_STAGE_3="anyres-256-1560-858k"

LLM_CKPT_DIR=/home/guozonghao/user/songhaolin/CKPT/Qwen2-7B-Instruct
CLIP_CKPT_DIR=/mnt/nfs_200T/optics/SHL/CKPT/moonvit-so-400m-4-18-se-hirope2d-p10
projector=mlp

BASE_RUN_NAME_STAGE_1="$MODEL_SETTING-$DATA_SETTING_STAGE_1"
BASE_RUN_NAME_STAGE_2="$MODEL_SETTING-$DATA_SETTING_STAGE_2"
BASE_RUN_NAME_STAGE_3="$MODEL_SETTING-$DATA_SETTING_STAGE_3"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME_STAGE_1}"

ACCELERATE_DISABLE_NUMA_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version qwen_2 \
    --model_mode='uhd_v2_5' \
    --data_path /mnt/nfs_200T/optics/SHL/Datasets/llava-uhd-2.5-data/stage-2-4M/stage-2-4M-558k.yaml \
    --image_folder /mnt/nfs_200T/optics/SHL/Datasets/llava-uhd-2.5-data/stage-2-4M/images/ \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type $projector \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/guozonghao/user/songhaolin/CKPT/stage_1/${BASE_RUN_NAME_STAGE_1} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
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

############### OCR+图文交错 ################
LLM_CKPT_DIR="/home/zhangyichen/users/zhangyichen/checkpoints/stage_1/${BASE_RUN_NAME_STAGE_1}"

DONE_FLAG="$LLM_CKPT_DIR/done.flag"

if [ "$RANK" == "0" ]; then
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
fi

# 多机同步等待 done.flag
while [ ! -f "$DONE_FLAG" ]; do
    sleep 5
done

echo "BASE_RUN_NAME: ${BASE_RUN_NAME_STAGE_2}"

ACCELERATE_DISABLE_NUMA_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py\
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version qwen_2 \
    --model_mode='uhd_v2_5' \
    # --data_path /home/guozonghao/user/sunshichu/data/llava-uhd-2.5-data/LLaVA-SFT-858K/json_files/llava_new_replace_text-new.json \
    # --image_folder /home/guozonghao/user/sunshichu/data/llava-uhd-2.5-data/LLaVA-SFT-858K/images \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type $projector \
    --mm_tunable_parts="mm_vision_tower,mm_language_model,mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/guozonghao/user/songhaolin/CKPT/stage_2/${BASE_RUN_NAME_STAGE_2} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --split_patch_size 40   # patch_size * 压缩比例

############### SFT ################
LLM_CKPT_DIR="/home/guozonghao/user/songhaolin/CKPT/stage_2/${BASE_RUN_NAME_STAGE_2}"

DONE_FLAG="$LLM_CKPT_DIR/done.flag"

if [ "$RANK" == "0" ]; then
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
fi

# 多机同步等待 done.flag
while [ ! -f "$DONE_FLAG" ]; do
    sleep 5
done

echo "BASE_RUN_NAME: ${BASE_RUN_NAME_STAGE_3}"

ACCELERATE_DISABLE_NUMA_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py\
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version qwen3 \
    --model_mode='uhd_v2_5' \
    --data_path /home/guozonghao/user/sunshichu/data/llava-uhd-2.5-data/LLaVA-SFT-858K/json_files/llava_new_replace_text-new.json \
    --image_folder /home/guozonghao/user/sunshichu/data/llava-uhd-2.5-data/LLaVA-SFT-858K/images \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type $projector \
    --mm_tunable_parts="mm_vision_tower,mm_language_model,mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/guozonghao/user/songhaolin/CKPT/stage_3/${BASE_RUN_NAME_STAGE_3} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --run_name ${BASE_RUN_NAME_STAGE_3} \
    --attn_implementation flash_attention_2 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --single True \
    --resolution 1560 \
    --any_res True \
    --split_patch_size 40   # patch_size * 压缩比例