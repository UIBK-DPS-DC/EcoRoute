#!/bin/bash

set -ex

if [ -z "$MODEL" ]; then
    echo "MODEL env var not set"
    exit 1
fi

if [ -z "$MODEL_FILE" ]; then
    echo "MODEL_FILE env var not set"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "Starting at $(date)"

echo "MODEL=$MODEL"
echo "MODEL_FILE=$MODEL_FILE"

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh


echo "Creating venv..."
python3 -m venv venv
source venv/bin/activate

echo "Installing huggingface_hub..."
pip install huggingface_hub

echo "Downloading model..."
hf download $MODEL $MODEL_FILE --local-dir .

echo "Creating Modelfile..."
echo -e "FROM ./$MODEL_FILE\n\nSYSTEM You are a helpful assistant." > Modelfile

echo "Creating Ollama model..."
ollama create $MODEL -f Modelfile

echo "Done."