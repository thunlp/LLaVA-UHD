from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="llava-uhd",
    version="2",
    description="https://github.com/thunlp/LLaVA-UHD",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "torch==2.1.2",
        "torchvision==0.16.2",
        "huggingface-hub==0.24.6",
        "transformers==4.37.2",
        "tokenizers==0.15.1",
        "sentencepiece==0.1.99",
        "shortuuid",
        "accelerate==0.21.0",
        "peft",
        "bitsandbytes",
        "pydantic",
        "markdown2[all]",
        "numpy",
        "scikit-learn==1.2.2",
        "gradio==4.16.0",
        "gradio_client==0.8.1",
        "requests",
        "httpx==0.24.0",
        "uvicorn",
        "fastapi",
        "einops==0.6.1",
        "einops-exts==0.0.4",
        "timm==0.6.13",
        "kornia",
        "omegaconf",
        "pytorch-lightning",
        "tqdm",
        "torchmetrics",
        "matplotlib",
        "hydra-core",
        "memory_profiler"
    ],
    extras_require={
        "train": ["deepspeed==0.12.6", "ninja", "wandb"],
        "build": ["build", "twine"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=find_packages(exclude=["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]),
    url="https://github.com/thunlp/LLaVA-UHD",
    project_urls={
        "Bug Tracker": "https://github.com/thunlp/LLaVA-UHD",
    },
    ext_modules=[
        CUDAExtension(
            'adaptive_conv_cuda_impl',
            [
                'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
                'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
            ]),
        CppExtension(
            'adaptive_conv_cpp_impl',
            ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
            undef_macros=["NDEBUG"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)