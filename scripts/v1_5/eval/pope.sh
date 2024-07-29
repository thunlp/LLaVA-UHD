#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava_pope_test_my"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/pope/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 1 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/pope/answers/$SPLIT/$CKPT/merge_slice.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/pope/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/$SPLIT.jsonl \
    --result-file ./playground/data/eval/pope/answers/$SPLIT/$CKPT/merge_slice.jsonl


# CKPT="llava-v1.5-adapt"

# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints_new/$CKPT \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl
