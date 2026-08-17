#!/bin/bash

set -e

if [ -z "$MODEL" ]; then
    echo "MODEL env var not set"
    exit 1
fi

if [ -z "$MODEL_FILE" ]; then
    echo "MODEL_FILE env var not set"
    exit 1
fi

python3 -m venv venv
source venv/bin/activate
pip install huggingface_hub

hf download $MODEL $MODEL_FILE --local-dir .

echo -e "FROM ./$MODEL_FILE\n\nSYSTEM You are a helpful assistant." > Modelfile

ollama create $MODEL -f Modelfile

