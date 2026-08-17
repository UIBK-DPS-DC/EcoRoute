#!/usr/bin/env bash

#   "google/pegasus-xsum"
#   "google/pegasus-cnn_dailymail"
#   "meta-llama/Llama-3.2-1B-Instruct"
#   "meta-llama/Llama-3.1-8B-Instruct"

  # "Qwen/Qwen2.5-1.5B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2.5-Coder-3B-Instruct"
  # "Qwen/Qwen2.5-Coder-7B-Instruct"
  # "Qwen/Qwen2.5-0.5B-Instruct"
  # "Qwen/Qwen2.5-3B-Instruct"

  # "google/gemma-3-1b-it"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
# MODELS=(
#   "Qwen/Qwen2.5-1.5B-Instruct"
#   "Qwen/Qwen2.5-7B-Instruct"
#   "Qwen/Qwen2.5-Coder-3B-Instruct"
#   "Qwen/Qwen2.5-Coder-7B-Instruct"
#   "Qwen/Qwen2.5-0.5B-Instruct"
#   "Qwen/Qwen2.5-3B-Instruct"
#   "mistralai/Mistral-7B-Instruct-v0.3"
# )

MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-Coder-3B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)
# "google/gemma-4-E2B-it"
# "google/gemma-4-E4B-it"
#   "google/gemma-4-31B-it"

PORT=8000

for MODEL in "${MODELS[@]}"; do
  echo "========================================"
  echo "Starting model: $MODEL"
  echo "========================================"

  # Start vLLM server in background
  vllm serve "$MODEL" \
    --additional_config '{"SLO_limits_for_dynamic_batch":"50"}' \
    --max-num-seqs 1 \
    --max_num_batched_tokens 9000 \
    --max-model-len 9000 \
    --host localhost \
    --port $PORT \
    --dtype half \
    --trust-remote-code &

    # --block-size 64 \

  SERVER_PID=$!

  echo "Waiting for server to be ready..."

  # Wait until server responds
  until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 2
  done

  echo "Server ready. Running trace generation..."

  # python generate_traces.py "$MODEL" -n 50 -p "./hf_datasets/"
  # python generate_traces.py "$MODEL" -n 50 -s 50 -p "./hf_datasets/"
  # python generate_traces.py "$MODEL" -n 50 -s 100 -p "./hf_datasets/"
  # python generate_traces.py "$MODEL" -n 50 -s 150 -p "./hf_datasets/"
  # python generate_traces.py "$MODEL" -n 50 -s 200 -p "./hf_datasets/"

  for S in $(seq 150 50 1950); do
    echo "Running batch starting at $S"
    python generate_traces.py "$MODEL" -n 50 -s $S -p "./hf_datasets/"
  done

  echo "Finished model: $MODEL"

  # Kill the server
  kill $SERVER_PID
  wait $SERVER_PID 2>/dev/null

  echo "Server stopped."
done

echo "All models processed."