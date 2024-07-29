#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava_test_CQM-A"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/scienceqa/$SPLIT.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --num_beams 3 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/merge_slice.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file $output_file \
    --output-file ./playground/data/eval/scienceqa/answers/output_$CKPT.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/result_$CKPT.json



# python -m llava.eval.model_vqa_science \
#     --model-path ./checkpoints_new/$CKPT \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers/output_$CKPT.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers/result_$CKPT.json
