export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc-per-node=8 run.py --data OCRBench MMMU_DEV_VAL SEEDBench_IMG MMBench_TEST_EN RealWorldQA HRBench4K --model llava_uhd2 --verbose