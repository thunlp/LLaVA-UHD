#!/bin/bash


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="chartqa_questions"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/chartqa/$SPLIT.jsonl \
        --image-folder ./playground/data/ureader/ChartQA/test/png \
        --answers-file ./playground/data/eval/chartqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 1 \
        --conv-mode qwen_1_5 &
done

wait

output_file=./playground/data/eval/chartqa/answers/$SPLIT/$CKPT/merge_slice.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/chartqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_docvqa \
    --annotation-file ./playground/data/eval/chartqa/chartqa_annotations.jsonl \
    --result-file $output_file \
    --mid_result ./playground/data/eval/chartqa/mid_results/$CKPT.jsonl \
    --output_result ./playground/data/eval/chartqa/exp_results/$CKPT/chartqa_result.jsonl


# python -m mova.eval.model_vqa_loader \
#     --model-path checkpoints/mova-8b \
#     --question-file ./playground/data/eval/chartqa/test_all.jsonl \
#     --image-folder ./playground/data/eval/chartqa/images \
#     --answers-file ./playground/data/eval/chartqa/answers/mova-8b.jsonl \
#     --temperature 0 \
#     --conv-mode mova_llama3

# python ./playground/data/eval/chartqa/eval_chartqa.py \
#     --annotation-file ./playground/data/eval/chartqa/test_all.jsonl \
#     --result-file ./playground/data/eval/chartqa/answers/mova-8b.jsonl \
#     --mid_result ./playground/data/eval/chartqa/mid_results/mova-8b.jsonl \
#     --output_result ./playground/data/eval/chartqa/results/mova-8b.jsonl 
