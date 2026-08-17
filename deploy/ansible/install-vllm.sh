#!/bin/bash

set -e

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install vllm --torch-backend=cu124

uv pip uninstall transformers
uv pip install "transformers<5"

uv cache clean flashinfer-python
uv pip uninstall flashinfer-python

unset LD_LIBRARY_PATH