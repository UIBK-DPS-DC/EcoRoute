#!/bin/bash

set -e

if [ -z "$MODEL" ]; then
    echo "MODEL env var not set"
    exit 1
fi

if [ -z "$NATS" ]; then
    echo "NATS env var not set"
    exit 1
fi

if [ -z "$SITE" ]; then
    echo "SITE env var not set"
    exit 1
fi

screen -list | grep -q '\.vllm[[:space:]]' || screen -dmS "vllm"

screen -S "vllm" -x -X screen bash -c "MODEL=$MODEL\nNATS=$NATS\nSITE=$SITE\n./model.sh\nexec bash"

# ./model.sh &