#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-.uv_python}"

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .

python --version
python -c "import torch, transformers, trl, peft, datasets; print('alignment env ok')"

