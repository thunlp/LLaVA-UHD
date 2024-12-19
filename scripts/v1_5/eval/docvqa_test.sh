#!/bin/bash

python -m mova.eval.model_vqa_loader \
    --model-path checkpoints/mova-8b \
    --question-file ./playground/data/eval/docvqa/test.jsonl \
    --image-folder ./playground/data/eval/docvqa/test/documents/ \
    --answers-file ./playground/data/eval/docvqa/answers/mova-8b.jsonl \
    --temperature 0 \
    --conv-mode mova_llama3

python scripts/convert_docvqa_for_submission.py \
    --result-dir ./playground/data/eval/docvqa/answers \
    --upload_dir ./playground/data/eval/docvqa/upload_results \
    --experiment mova-8b
