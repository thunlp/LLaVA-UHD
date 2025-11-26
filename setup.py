from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="llava",
    version="1.7.0.dev0",
    description="LLaVA OneVision: The Next Generation of LLaVA with Better Image and Video Understanding Capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    url="https://llava-vl.github.io",
    project_urls={
        "Homepage": "https://llava-vl.github.io",
        "Bug Tracker": "https://github.com/haotian-liu/LLaVA/issues",
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
        "torch==2.1.2",
        "numpy<2.0",
    ],

    # ext_modules=[
    #     CUDAExtension(
    #         'adaptive_conv_cuda_impl',
    #         [
    #             'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
    #             'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
    #         ]),
    #     CppExtension(
    #         'adaptive_conv_cpp_impl',
    #         ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
    #         undef_macros=["NDEBUG"]),
    # ],
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
            "numpy<2.0",
            "open_clip_torch",
            "fastapi",
            "markdown2[all]",
            "requests",
            "sentencepiece",
            "torch==2.1.2",
            "torchvision",
            "uvicorn",
            "wandb==0.18.7",
            "deepspeed==0.14.4",
            "peft==0.4.0",
            "accelerate==0.29.3",
            "tokenizers<0.22",
            "transformers==4.51.0",
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
