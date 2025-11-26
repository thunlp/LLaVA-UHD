
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 统计 GPU 数量
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_IDS[@]}

export LMUData=/home/zhangyichen/LMUData/LMUData
echo $LMUData

# MODEL_NAME=llava-uhd-qwen3-moonvit-4x4pooling-858k-256px-1024px-858k
MODEL_NAME=llava-uhd-qwen3-vituhd-merger-858k-256px-2048px-858k--1.4upscale-sota-eval
MASTER_PORT=19505

ACCELERATE_CPU_AFFINITY=0 torchrun --master_port=$MASTER_PORT --nproc-per-node=$NUM_GPUS run.py --data \
    MME SEEDBench_IMG \
    MMStar POPE HallusionBench RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
    AI2D_TEST OCRBench TextVQA_VAL DocVQA_TEST ChartQA_TEST \
    --model $MODEL_NAME --verbose                

# ACCELERATE_CPU_AFFINITY=1 torchrun --master_port=$MASTER_PORT --nproc-per-node=$NUM_GPUS run.py --data \
#     CV-Bench-2D CV-Bench-3D \
#     --model $MODEL_NAME --verbose
# ACCELERATE_CPU_AFFINITY=1 torchrun --master_port=$MASTER_PORT --nproc-per-node=$NUM_GPUS run.py --data \
#     MME MMBench_DEV_EN_V11 SEEDBench_IMG MMVet MathVista_MINI \
#     MMStar POPE HallusionBench RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
#     AI2D_TEST OCRBench TextVQA_VAL DocVQA_TEST ChartQA_TEST \
#     --model $MODEL_NAME --mode infer --verbose                  

# ACCELERATE_CPU_AFFINITY=1 torchrun --master_port=$MASTER_PORT --nproc-per-node=$NUM_GPUS run.py --data \
#     MME MMBench_DEV_EN_V11 SEEDBench_IMG \
#     MMStar POPE HallusionBench RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
#     AI2D_TEST OCRBench TextVQA_VAL DocVQA_TEST ChartQA_TEST \
#     --model $MODEL_NAME --verbose
