#!/bin/bash
set -e  
set -o pipefail

# --- Conda init ---
if ! command -v conda &> /dev/null; then
    echo "❌ Conda unavailable. Please install Conda first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="LLaVA-UHD-v3"

echo "📦 pip install ..."
conda run -n "$ENV_NAME" pip install -q --upgrade pip
conda run -n "$ENV_NAME" pip install -q imgaug openpyxl
conda run -n "$ENV_NAME" pip install -q torch==2.1.2 torchvision==0.16.2 
conda run -n "$ENV_NAME" pip install -q "numpy>=2.0,<2.3"
conda run -n "$ENV_NAME" pip install -q -e ".[train]"
conda run -n "$ENV_NAME" pip install -q -e .
conda run -n "$ENV_NAME" pip install -q ./flash_attn-2.7.0.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

conda run -n "$ENV_NAME" wandb offline

conda activate $ENV_NAME
echo "🎉 '$ENV_NAME' install successfully!"
