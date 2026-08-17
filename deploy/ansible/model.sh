#!/bin/bash

# echo "LLM_ID=$MODEL" > .env
# echo "NATS_URL=$NATS" >> .env
# echo "SITE=$SITE" >> .env

cd code/inference
python3 -m venv llm-venv
source llm-venv/bin/activate

pip install -r requirements.txt

# cd inference
# python llm.py &