#!/bin/bash

set -e

if [ -z "$ROUTER" ]; then
    echo "ROUTER env var not set"
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

for session in $(screen -ls | grep -oP '\d+\.\w+' | cut -d. -f1); do screen -S "${session}" -X quit; done

cd code

screen -list | grep -q '\.router[[:space:]]' || screen -dmS "router"

screen -S "router" -x -X screen bash -c "sudo docker compose up $NATS";
screen -S "router" -x -X screen bash -c "sudo docker compose up $ROUTER --build"
