#!/bin/bash


# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# # multiple evaluation
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava_test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/vizwiz/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vizwiz/test \
        --answers-file ./playground/data/eval/vizwiz/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vizwiz/answers/$SPLIT/$CKPT/merge_slice.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vizwiz/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/$SPLIT.jsonl \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$CKPT.json
