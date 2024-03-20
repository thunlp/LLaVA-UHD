<div align="center">
  
# LLaVA-UHD

**A Large Multimodal Model Perceiving Any Aspect Ratio and High-Resolution Images**
</div>

This repository hosts the code, data, and model weight of **LLaVA-UHD**, a novel framework that enables Large Multimodal Models (LMMs) to efficiently perceive images in any aspect ratio and high resolution.

Notably, our model built on LLaVA-1.5 336×336 supports 6 times
larger (i.e., 672×1088) resolution images using only 94% inference computation,
and achieves 6.4 accuracy improvement on TextVQA. Moreover, the model can be efficiently trained in academic settings, within 23 hours on 8 A100 GPUs (vs. 26 hours of LLaVA-1.5). Visit our 📃 [paper](https://arxiv.org/pdf/2403.11703.pdf) here!


## Overview

![The LLaVA-UHD framework](LLaVA-UHD.jpg)

LLaVA-UHD includes three key components to deal with native-resolution images: 

-  An image modularization strategy that divides native-resolution images into smaller variable-sized
slices for efficient and extensible encoding.

-  A compression module that further
condenses image tokens from visual encoders.

-  A spatial schema to organize
slice tokens for LLMs. Comprehensive experiments show that LLaVA-UHD out-
performs established LMMs trained with 2-3 orders of magnitude more data on
9 benchmarks. 

## Preparing
To reproduce the results of the paper, please set up the Python environment using the following code:
```bash
conda create -n llava-uhd python=3.10
conda activate llava-uhd
pip install -r requirements.txt
```

## Pretraining Code
Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

You should refer to the documentation of llava1.5, set up the environment according to llava1.5's way, and organize the training data properly, placing it in the path ./playground. Then run the following code for inference:

```bash
bash scripts/pretrain.sh
```

## Fine-tuning Code

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:
- COCO: train2017
- GQA: images
- OCR-VQA: download script, we save all files as .jpg
- TextVQA: train_val_images
- VisualGenome: part1, part2

Download dataset images as in the finetuning process of llava1.5, place them in the playground, and then run the following code:
```bash
bash scripts/finetune.sh
```

## Evaluation Code

When evaluating the model, we almost synchronously use the testing code of llava1.5, and the basic usage method is consistent. Please refer to [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#evaluation) for help. We provide the same script to complete the testing.

## Exploratory experiments
Exploratory experiments relevant to the paper and more reliable code are currently being further organized. Some of the code is not yet in its final version. Stay tuned!

## Citation

If you find LLaVA-UHD useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{xu2024llava-uhd,
  title={{LLaVA-UHD}: an LMM Perceiving Any Aspect Ratio and High-Resolution Images},
  author={Xu, Ruyi and Yao, Yuan and Guo, Zonghao and Cui, Junbo and Ni, Zanlin and Ge, Chunjiang and Chua, Tat-Seng and Liu, Zhiyuan and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11703},
  year={2024}
}
```
