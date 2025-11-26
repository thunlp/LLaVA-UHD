

# export OPENAI_API_KEY=sk-NTcOlRNTJTCoc31NFfA3E43f68Fe4f53819a9e925bCa22Cc
export OPENAI_API_KEY=sk-kYJ2x8TKGQoH4eBQDUdSumvhVIDgpZzIxldChxqR9OPUAMzL
export OPENAI_API_BASE=https://yeysai.com/v1
export LMUData=/home/zhangyichen/LMUData/LMUData
echo $LMUData

MODEL_NAME=GPT4o_MINI

python run.py --data \
    CV-Bench-2D CV-Bench-3D \
    --model $MODEL_NAME --verbose --api-nproc 128

python run.py --data \
    CV-Bench-2D CV-Bench-3D \
    --model $MODEL_NAME --verbose --api-nproc 128

# python run.py --data \
#     MME MMBench_DEV_EN_V11 SEEDBench_IMG MMVet MathVista_MINI \
#     MMStar POPE HallusionBench RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
#     AI2D_TEST OCRBench TextVQA_VAL DocVQA_TEST ChartQA_TEST \
#     CV-Bench-2D CV-Bench-3D \
#     --model $MODEL_NAME --mode infer --verbose --api-nproc 128

# python run.py --data \
#     MME MMBench_DEV_EN_V11 SEEDBench_IMG \
#     MMStar POPE HallusionBench RealWorldQA MMMU_DEV_VAL ScienceQA_TEST  \
#     AI2D_TEST OCRBench TextVQA_VAL DocVQA_TEST ChartQA_TEST \
#     CV-Bench-2D CV-Bench-3D \
#     --model $MODEL_NAME --verbose

# python run.py --data \
#     MMVet MathVista_MINI\
#     --model $MODEL_NAME --api-nproc 128