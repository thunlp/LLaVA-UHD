

# python -m llava.eval.model_vqa_loader \
#     --model-path checkpoints/mova-8b \
#     --question-file ./playground/data/eval/ai2d/test.jsonl \
#     --image-folder ./playground/data/eval/ai2d \
#     --answers-file ./playground/data/eval/ai2d/answers/mova-8b.jsonl \
#     --temperature 0 \
#     --conv-mode mova_llama3

#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/ai2d/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/ai2d/ai2d_images \
        --answers-file ./playground/data/eval/ai2d/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 1 \
        --conv-mode qwen_1_5 &
done

wait

output_file=./playground/data/eval/ai2d/answers/$SPLIT/$CKPT/merge_slice.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/ai2d/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_ai2d \
    --annotation-file ./playground/data/eval/ai2d/test_from_mova.jsonl \
    --result-file $output_file \
    --mid_result ./playground/data/eval/ai2d/mid_results/$CKPT.jsonl \
    --output_result ./exp_results/$CKPT/ai2d_result.jsonl