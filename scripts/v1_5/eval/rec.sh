# #!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
echo $CKPT
SPLIT="llava_ref3_test_2017"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints_new/$CKPT \
        --question-file ./playground/data/eval/rec/$SPLIT.jsonl \
        --image-folder ./playground/data/coco/train2017 \
        --answers-file ./playground/data/eval/rec/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/rec/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/rec/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_rec \
    --annotation-file ./playground/data/eval/rec/llava_ref3_labels.jsonl \
    --question-file ./playground/data/eval/rec/$SPLIT.jsonl \
    --result-file ./playground/data/eval/rec/answers/$SPLIT/$CKPT/merge.jsonl