#!/bin/bash
#parse options
while getopts 'n:j:s:p:f:x:y:' OPTION; do
  case "$OPTION" in
    n)
      JOB_NAME=$OPTARG ;;
    j)
      JBU_CKPT=$OPTARG ;;
    s)
      SCALE=$OPTARG ;;
    p)
      PRETRAIN_LR=$OPTARG ;;
    f)
      FINETUNE_LR=$OPTARG ;;
    x)
      PRBATCH=$OPTARG ;;
    y)
      FTRBATCH=$OPTARG ;;
    ?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

echo "JOB_NAME: $JOB_NAME"
echo "JBU_CKPT: $JBU_CKPT"
echo "SCALE: $SCALE"
echo "PRETRAIN_LR: $PRETRAIN_LR"
echo "FINETUNE_LR: $FINETUNE_LR"
echo "PRBATCH: $PRBATCH"
echo "FTRBATCH: $FTRBATCH"

wandb offline

CKPT=llava-uhd-144-7b
mkdir -p /data/checkpoints/$JOB_NAME/checkpoints_new/$CKPT
OUTPUT_DIR=/data/checkpoints/$JOB_NAME/checkpoints_new/$CKPT
LLM_CKPT_DIR=./pretrained_models/vicuna-7b-v1.5
CLIP_CKPT_DIR=./pretrained_models/clip-vit-large-patch14-336

echo $OUTPUT_DIR

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12345}

DISTRIBUTED_ARGS="
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ENDPOINT \
  --master_port $MASTER_PORT "

echo $DISTRIBUTED_ARGS


#pretrain script
#total batch size == 256

torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version plain \
    --feature_mode 'featup_muti_res' \
    --data_path $PWD/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $PWD/playground/data/LLaVA-Pretrain/images \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type adapt_spatial_resampler \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PRBATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCU_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --single True \
    --jbu_ckpt $JBU_CKPT \
    --feature_scale_mask $SCALE \
    --sft_encoder False

#full ft script
#total batch size == 128
ACCU_STEPS=1

torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version v1 \
    --feature_mode 'featup_muti_res' \
    --data_path ./llava_new_replace_text-new.json \
    --image_folder ./llava_new \
    --vision_tower $CLIP_CKPT_DIR \
    --pretrain_mm_mlp_adapter $OUTPUT_DIR/mm_projector.bin \
    --mm_projector_type adapt_spatial_resampler \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $FTRBATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 600 \
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
    --jbu_ckpt $JBU_CKPT \
    --report_to wandb \
    --sft_encoder True

# evaluation
pip install editdistance
sh eval.sh $CKPT