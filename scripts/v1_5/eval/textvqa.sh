#!/bin/bash


# # multiple evaluation
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava_textvqa_val_v051_ocr"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/textvqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/merge_slice.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/merge_slice.jsonl

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
