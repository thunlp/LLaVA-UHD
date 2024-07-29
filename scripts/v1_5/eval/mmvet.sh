#!/bin/bash


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava-mm-vet"

# python -m llava.eval.model_vqa \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/mm-vet/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/mm-vet/images \
        --answers-file ./playground/data/eval/mm-vet/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 5 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/mm-vet/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mm-vet/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $output_file \
    --dst ./playground/data/eval/mm-vet/results/$CKPT.json

