from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="llava-uhd",
    version="2",
    description="https://github.com/thunlp/LLaVA-UHD",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    url="https://github.com/thunlp/LLaVA-UHD",
    project_urls={
        "Bug Tracker": "https://github.com/thunlp/LLaVA-UHD",
    },
    packages=find_packages(
        include=["llava*", "trl*"],
        exclude=[
            "assets*",
            "benchmark*",
            "docs",
            "dist*",
            "playground*",
            "scripts*",
            "tests*",
            "checkpoints*",
            "project_checkpoints*",
            "debug_checkpoints*",
            "mlx_configs*",
            "wandb*",
            "notebooks*",
        ],
    ),

    install_requires=[
        "torch",
        "numpy",
    ],

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
    },

    extras_require={
        "standalone": [
            "shortuuid",
            "httpx==0.24.0",
            "einops",
            "ftfy",
        ],
        "train": [
            "llava[standalone]",
            "pynvml==11.5.0",
            "numpy",
            "open_clip_torch",
            "fastapi",
            "markdown2[all]",
            "requests",
            "sentencepiece",
            "torch==2.1.2",
            "torchvision==0.16.2",
            "uvicorn",
            "wandb==0.18.7",
            "deepspeed==0.14.4",
            "peft==0.4.0",
            "accelerate==0.29.3",
            "tokenizers==0.19",
            "transformers==4.40.1",
            "bitsandbytes",
            "scikit-learn==1.2.2",
            "sentencepiece~=0.1.99",
            "einops==0.6.1",
            "einops-exts==0.0.4",
            "gradio_client==0.2.9",
            "urllib3<=2.0.0",
            "datasets==2.16.1",
            "pydantic==1.10.8",
            "timm",
            "hf_transfer",
            "opencv-python",
            "av",
            "decord",
            "tyro",
            "scipy",
            'kornia',
            'omegaconf',
            'pytorch-lightning',
            'tqdm',
            'torchmetrics',
            'matplotlib',
            'hydra-core',
            'memory_profiler'
        ],
    },
    include_package_data=True,
)
