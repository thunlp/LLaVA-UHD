
#parse options
while getopts 'n:j:s:p:f:x:y:' OPTION; do
  case "$OPTION" in
    n)
      JOB_NAME=$OPTARG ;;
    j)
      VDIM_CKPT=$OPTARG ;;
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
echo "VDIM_CKPT: $VDIM_CKPT"
echo "SCALE: $SCALE"
echo "PRETRAIN_LR: $PRETRAIN_LR"
echo "FINETUNE_LR: $FINETUNE_LR"
echo "PRBATCH: $PRBATCH"
echo "FTRBATCH: $FTRBATCH"

#install
pip install imgaug --retries 10
pip install openpyxl --retries 10

# pip install --upgrade pip  # enable PEP 660 support
pip install -e . --retries 10

pip install -e ".[train]" --retries 10
pip install flash-attn --no-build-isolation --retries 10

wandb offline

CKPT=llava-uhd-144-13b
mkdir -p /data/checkpoints/$JOB_NAME/checkpoints_new/$CKPT
OUTPUT_DIR=/data/checkpoints/$JOB_NAME/checkpoints_new/$CKPT
LLM_CKPT_DIR=./pretrained_models/model_checkpoints/vicuna-13b-v1.5
CLIP_CKPT_DIR=./pretrained_models/models--openai--clip-vit-large-patch14-336

#full ft script
#FTRBATCH=4
# 多机多卡使用这种方式
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


#total batch size == 256
ACCU_STEPS=1

ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version plain \
    --model_mode='uhd_v2' \
    --data_path $PWD/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder $PWD/playground/data/LLaVA-Pretrain/images \
    --vision_tower $CLIP_CKPT_DIR \
    --mm_projector_type adapt_spatial_resampler_v2 \
    --mm_tunable_parts="mm_mlp_adapter" \
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
    --learning_rate $PRETRAIN_LR \
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
    --run_name $JOB_NAME \
    --attn_implementation flash_attention_2 \
    --single True \
    --vdim_ckpt $VDIM_CKPT \
    --feature_scale_mask 5 \
    --dataloader_drop_last True \
    --sft_vdim False

#full ft script
#total batch size == 128

ACCELERATE_CPU_AFFINITY=1 torchrun $DISTRIBUTED_ARGS llava/train/train_mem.py \
    --deepspeed $PWD/scripts/zero2_old.json \
    --model_name_or_path $LLM_CKPT_DIR \
    --version v1 \
    --model_mode='uhd_v2' \
    --data_path ./llava-uhdv2-sft.json \
    --image_folder ./llava_new \
    --vision_tower $CLIP_CKPT_DIR \
    --pretrain_mm_mlp_adapter ./mm_projector.bin \
    --mm_projector_type adapt_spatial_resampler_v2 \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $FTRBATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCU_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate $FINETUNE_LR \
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
    --run_name $JOB_NAME \
    --attn_implementation flash_attention_2 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --single False \
    --vdim_ckpt $VDIM_CKPT \
    --feature_scale_mask 5 \
    --sft_vdim True
