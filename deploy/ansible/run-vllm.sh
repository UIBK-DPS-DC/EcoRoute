#!/bin/bash

set -e

if [ -z "$MODEL" ]; then
    echo "MODEL env var not set"
    exit 1
fi

screen -list | grep -q '\.vllm[[:space:]]' || screen -dmS "vllm"

screen -S "vllm" -X stuff "MODEL=$MODEL\n./vllm.sh\n";

# ./vllm.sh &