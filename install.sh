#install
pip install imgaug
pip install openpyxl

pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.1.2
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation
