#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava_mme"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/MME/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 5 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/MME/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/MME/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


cp $output_file ./playground/data/eval/MME/answers/$CKPT.jsonl

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT




# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints_new/$CKPT \
#     --question-file ./playground/data/eval/MME/$SPLIT.jsonl \
#     --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
#     --temperature 0 \
#     --num_beams 3 \
#     --conv-mode vicuna_v1
