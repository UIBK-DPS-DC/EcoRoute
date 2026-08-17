#!/bin/bash

set -e

if [ -z "$MODEL" ]; then
    echo "MODEL env var not set"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN env var not set"
    exit 1
fi

source .venv/bin/activate

unset LD_LIBRARY_PATH
SLO_LITMIT=50

vllm serve $MODEL \
    --additional_config '{"SLO_limits_for_dynamic_batch":'${SLO_LITMIT}'}' \
    --max-num-seqs 256 \
    --block-size 32 \
    --max_num_batched_tokens 9000 \
    --max-model-len 9000 \
    --host localhost \
    --port 8000 \
    --dtype=half \
    --trust-remote-code

