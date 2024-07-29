#!/bin/bash

# SPLIT="mmbench_dev_cn_20231003"


# python -m llava.eval.model_vqa_mmbench \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/llava-v1.5-13b.jsonl \
#     --lang cn \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="mmbench_dev_cn_20231003"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --lang cn \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.2 \
        --num_beams 1 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment merge




