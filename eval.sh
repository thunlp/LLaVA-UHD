#!/bin/bash

# mkdir -p "./exp_results/$1"
# echo 'made a dir ./exp_results/'$1

NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/mme.sh $1 #> ./exp_results/$1/mme_result.log
echo 'mme done'

CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/ai2d.sh $1 #> ./exp_results/$1/ai2d_result.log
echo 'ai2d done'

CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/docvqa_val.sh $1 #> ./exp_results/$1/docvqa_eval_result.log
echo 'doc done'

CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/chartqa.sh $1 #> ./exp_results/$1/chartqa_result.log
echo 'chart done'

# traditional 
CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/textvqa.sh $1 #> ./exp_results/$1/textvqa_result.log
echo 'textvqa done'

CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/gqa.sh $1 #> ./exp_results/$1/gqa_result.log
echo 'gqa done'

CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES bash scripts/v1_5/eval/sqa.sh $1 #> ./exp_results/$1/scienceqa_result.log
echo 'sqa done'

echo 'All eval done, exiting successfully.'